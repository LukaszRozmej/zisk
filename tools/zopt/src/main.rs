//! zopt: ZisK ROM post-pass optimizer.
//! RISC-V ELF -> stock transpiler -> ZiskRom -> rewrites that only the ZisK ISA can express -> ziskbin ELF.
//!
//! usage: zopt <in.elf> <out.elf> [passes=jt,cf,af] [max_rounds]

use std::collections::{BTreeMap, HashMap, HashSet};
use zisk_core::{
    zisk_ops::{OpType, ZiskOp},
    ziskbin, ZiskInst, ZiskInstBuilder, ZiskRom, SRC_C, SRC_IMM, SRC_IND, SRC_REG, STORE_IND,
    STORE_NONE, STORE_REG,
};

const ALL: u64 = (1u64 << 40) - 2; // r1..r39

fn imm_a(i: &ZiskInst) -> u64 {
    i.a_offset_imm0 | (i.a_use_sp_imm1 << 32)
}
fn imm_b(i: &ZiskInst) -> u64 {
    i.b_offset_imm0 | (i.b_use_sp_imm1 << 32)
}
fn set_imm_a(i: &mut ZiskInst, v: u64) {
    i.a_src = SRC_IMM;
    i.a_offset_imm0 = v & 0xffff_ffff;
    i.a_use_sp_imm1 = v >> 32;
}
fn set_imm_b(i: &mut ZiskInst, v: u64) {
    i.b_src = SRC_IMM;
    i.b_offset_imm0 = v & 0xffff_ffff;
    i.b_use_sp_imm1 = v >> 32;
}

fn reads(i: &ZiskInst) -> u64 {
    let mut m = 0;
    if i.a_src == SRC_REG {
        m |= 1 << i.a_offset_imm0;
    }
    if i.b_src == SRC_REG {
        m |= 1 << i.b_offset_imm0;
    }
    m
}
fn writes(i: &ZiskInst) -> u64 {
    if i.store == STORE_REG {
        1 << i.store_offset
    } else {
        0
    }
}
fn op_type(i: &ZiskInst) -> OpType {
    ZiskOp::try_from_code(i.op).map(|o| o.op_type()).unwrap_or(OpType::Internal)
}
/// Ops whose only effect is (c, flag) = f(a, b): no precompile memory, no fcall state.
fn is_pure(i: &ZiskInst) -> bool {
    !i.is_precompiled
        && matches!(
            op_type(i),
            OpType::Internal
                | OpType::Binary
                | OpType::BinaryE
                | OpType::ArithAm32
                | OpType::ArithA32
        )
}
/// Plain control: next pc is pc+j1 (flag) or pc+j2, no pc capture, no end.
fn is_plain(i: &ZiskInst) -> bool {
    is_pure(i) && !i.set_pc && !i.store_pc && !i.end
}
fn is_pure_jump(i: &ZiskInst) -> bool {
    i.op == ZiskOp::FLAG
        && i.a_src == SRC_IMM
        && i.b_src == SRC_IMM
        && i.store == STORE_NONE
        && !i.store_pc
        && !i.set_pc
        && !i.end
}

struct Rom {
    addrs: Vec<u64>,
    idx: HashMap<u64, usize>,
}

fn successors(i: &ZiskInst) -> Option<([u64; 2], usize)> {
    if i.set_pc {
        return None;
    }
    if i.end {
        return Some(([0; 2], 0));
    }
    let pc = i.paddr as i64;
    if i.is_precompiled || !is_pure(i) {
        // external ops never raise the flag; j1 may carry an extended argument
        return Some(([(pc + i.jmp_offset2) as u64, 0], 1));
    }
    Some((
        [(pc + i.jmp_offset1) as u64, (pc + i.jmp_offset2) as u64],
        if i.jmp_offset1 == i.jmp_offset2 { 1 } else { 2 },
    ))
}

/// Backward liveness; returns live-out per instruction.
fn liveness(insts: &BTreeMap<u64, ZiskInstBuilder>, r: &Rom) -> Vec<u64> {
    let n = r.addrs.len();
    let mut succ = Vec::with_capacity(n);
    let mut preds: Vec<Vec<usize>> = vec![vec![]; n];
    for (k, a) in r.addrs.iter().enumerate() {
        let i = &insts[a].i;
        let s = successors(i).and_then(|(targets, len)| {
            let mut indices = [0; 2];
            for j in 0..len {
                indices[j] = *r.idx.get(&targets[j])?;
            }
            Some((indices, len))
        });
        if let Some((targets, len)) = &s {
            for &t in &targets[..*len] {
                preds[t].push(k);
            }
        }
        succ.push(s);
    }
    let rd: Vec<u64> = r.addrs.iter().map(|a| reads(&insts[a].i)).collect();
    let wr: Vec<u64> = r.addrs.iter().map(|a| writes(&insts[a].i)).collect();
    let mut live_in = vec![0u64; n];
    let mut live_out = vec![0u64; n];
    let mut work: Vec<usize> = (0..n).collect();
    let mut queued = vec![true; n];
    while let Some(k) = work.pop() {
        queued[k] = false;
        let out = match &succ[k] {
            None => ALL,
            Some((targets, len)) => targets[..*len].iter().fold(0, |m, &t| m | live_in[t]),
        };
        live_out[k] = out;
        let inn = rd[k] | (out & !wr[k]);
        if inn != live_in[k] {
            live_in[k] = inn;
            for &p in &preds[k] {
                if !queued[p] {
                    queued[p] = true;
                    work.push(p);
                }
            }
        }
    }
    live_out
}

#[derive(Default, Debug)]
struct Stats {
    jt: usize,
    cf: usize,
    cf_const: usize,
    af: usize,
    lj: usize,
}

fn index(insts: &BTreeMap<u64, ZiskInstBuilder>) -> Rom {
    let addrs: Vec<u64> = insts.keys().copied().collect();
    let idx = addrs.iter().enumerate().map(|(k, a)| (*a, k)).collect();
    Rom { addrs, idx }
}

/// Jump threading: retarget any plain jump/branch slot that lands on a pure jump.
fn pass_jt(insts: &mut BTreeMap<u64, ZiskInstBuilder>, st: &mut Stats) {
    let final_target = |insts: &BTreeMap<u64, ZiskInstBuilder>, mut t: u64| -> u64 {
        for _ in 0..16 {
            match insts.get(&t) {
                Some(u) if is_pure_jump(&u.i) => {
                    let nt = (t as i64 + u.i.jmp_offset1) as u64;
                    // the instruction after the jump must not read the jump's c
                    match insts.get(&nt) {
                        Some(v) if v.i.a_src != SRC_C && v.i.b_src != SRC_C && nt != t => t = nt,
                        _ => break,
                    }
                }
                _ => break,
            }
        }
        t
    };
    let addrs: Vec<u64> = insts.keys().copied().collect();
    for a in addrs {
        let i = &insts[&a].i;
        if !is_plain(i) && !is_pure_jump(i) {
            continue;
        }
        let (j1, j2) = (i.jmp_offset1, i.jmp_offset2);
        let t1 = final_target(insts, (a as i64 + j1) as u64);
        let t2 = final_target(insts, (a as i64 + j2) as u64);
        let (n1, n2) = (t1 as i64 - a as i64, t2 as i64 - a as i64);
        if n1 != j1 || n2 != j2 {
            let m = &mut insts.get_mut(&a).unwrap().i;
            m.jmp_offset1 = n1;
            m.jmp_offset2 = n2;
            st.jt += 1;
        }
    }
}

/// Merge X (single fall-through successor Y) into one instruction placed at X when X's only
/// effect is a register that dies at Y. Covers constant folding (X = copyb imm) and address
/// folding (X = add reg, imm feeding Y's indirect base). Y stays in place for its other
/// predecessors.
fn pass_fold(insts: &mut BTreeMap<u64, ZiskInstBuilder>, cf: bool, af: bool, st: &mut Stats) {
    let r = index(insts);
    let live_out = liveness(insts, &r);
    let mut new: Vec<(u64, ZiskInst)> = vec![];
    let mut touched: HashSet<u64> = Default::default();
    for &a in &r.addrs {
        let x = &insts[&a].i;
        if a & 1 != 0
            || x.next_internal_inst.is_some()
            || !is_plain(x)
            || x.jmp_offset1 != x.jmp_offset2
        {
            continue;
        }
        if x.store != STORE_REG || x.a_src == SRC_C || x.b_src == SRC_C || x.b_src == SRC_IND {
            continue;
        }
        let d = x.jmp_offset2;
        let ya = (a as i64 + d) as u64;
        let Some(&yk) = r.idx.get(&ya) else { continue };
        if touched.contains(&a) || touched.contains(&ya) {
            continue;
        }
        let y = &insts[&ya].i;
        if y.next_internal_inst.is_some()
            || y.set_pc
            || y.store_pc
            || y.end
            || y.is_precompiled
            || !is_pure(y)
        {
            continue;
        }
        if y.a_src == SRC_C || y.b_src == SRC_C {
            continue;
        }
        let rx = x.store_offset as u64;
        let rbit = 1u64 << rx;
        if reads(y) & rbit == 0 {
            continue;
        }
        let dead = live_out[yk] & rbit == 0 || writes(y) & rbit != 0;
        if !dead {
            continue;
        }
        let mut f = y.clone();
        f.paddr = a;
        // The ROM trace places each instruction at row `index`: the merged one takes X's row, or two
        // instructions share Y's and the lookups at X's address find no ROM row.
        f.index = x.index;
        f.jmp_offset1 += d;
        f.jmp_offset2 += d;
        f.riscv_inst = None;
        f.verbose = String::new();
        let uses_ind_a = y.b_src == SRC_IND || y.store == STORE_IND;
        let mut ok = false;
        if cf && x.op == ZiskOp::COPYB && x.b_src == SRC_IMM {
            let v = imm_b(x);
            let a_is_r = y.a_src == SRC_REG && y.a_offset_imm0 == rx;
            let b_is_r = y.b_src == SRC_REG && y.b_offset_imm0 == rx;
            if !(a_is_r && uses_ind_a && v >> 32 != 0) {
                if a_is_r {
                    set_imm_a(&mut f, v);
                }
                if b_is_r {
                    set_imm_b(&mut f, v);
                }
                ok = true;
                st.cf += 1;
                // both operands constant: evaluate at build time when the flag cannot matter
                if f.a_src == SRC_IMM
                    && f.b_src == SRC_IMM
                    && f.op != ZiskOp::FLAG
                    && !uses_ind_a
                    && f.jmp_offset1 == f.jmp_offset2
                    && is_pure(&f)
                {
                    let (c, _) = ZiskOp::execute(f.op, imm_a(&f), imm_b(&f));
                    let mut zib = ZiskInstBuilder { i: f.clone() };
                    zib.op("copyb").unwrap();
                    f = zib.i;
                    set_imm_a(&mut f, 0);
                    set_imm_b(&mut f, c);
                    st.cf_const += 1;
                }
            }
        } else if af
            && x.op == ZiskOp::ADD
            // Replacing a must preserve the operation's result and flag, not just its address.
            && matches!(
                y.op,
                ZiskOp::COPYB | ZiskOp::SIGNEXTEND_B | ZiskOp::SIGNEXTEND_H | ZiskOp::SIGNEXTEND_W
            )
            && x.a_src == SRC_REG
            && x.b_src == SRC_IMM
            && uses_ind_a
            && y.a_src == SRC_REG
            && y.a_offset_imm0 == rx
            && !(y.b_src == SRC_REG && y.b_offset_imm0 == rx)
        {
            let k = imm_b(x) as i64;
            if (-(1i64 << 31)..(1i64 << 31)).contains(&k) {
                f.a_offset_imm0 = x.a_offset_imm0;
                // offsets are 64-bit two's complement, as the transpiler encodes them
                if y.b_src == SRC_IND {
                    f.b_offset_imm0 = (y.b_offset_imm0 as i64).wrapping_add(k) as u64;
                }
                if y.store == STORE_IND {
                    f.store_offset = y.store_offset.wrapping_add(k);
                }
                ok = true;
                st.af += 1;
            }
        }
        if ok {
            touched.insert(a);
            touched.insert(ya);
            new.push((a, f));
        }
    }
    for (a, f) in new {
        insts.get_mut(&a).unwrap().i = f;
    }
}

/// Load + indirect jump: `ld t, k(base); jr t` (jalr rd=0, even imm) becomes one instruction at
/// the load that writes t and jumps to it. Drops jalr's `& ~1`: exact when loaded targets are even.
fn pass_lj(insts: &mut BTreeMap<u64, ZiskInstBuilder>, st: &mut Stats) {
    let addrs: Vec<u64> = insts.keys().copied().collect();
    let mut new = vec![];
    for a in addrs {
        let x = &insts[&a].i;
        if a & 1 != 0
            || x.next_internal_inst.is_some()
            || !is_plain(x)
            || x.jmp_offset1 != x.jmp_offset2
        {
            continue;
        }
        if x.op != ZiskOp::COPYB
            || x.b_src != SRC_IND
            || x.ind_width != 8
            || x.store != STORE_REG
            || x.a_src != SRC_REG
        {
            continue;
        }
        let ya = (a as i64 + x.jmp_offset2) as u64;
        let Some(y) = insts.get(&ya).map(|b| &b.i) else { continue };
        let t = x.store_offset as u64;
        let is_jr = y.set_pc
            && !y.store_pc
            && y.store == STORE_NONE
            && y.op == ZiskOp::AND
            && y.a_src == SRC_IMM
            && imm_a(y) == 0xffff_ffff_ffff_fffe
            && y.b_src == SRC_REG
            && y.b_offset_imm0 == t
            && y.next_internal_inst.is_none();
        if !is_jr {
            continue;
        }
        let mut f = x.clone();
        f.set_pc = true;
        f.jmp_offset1 = y.jmp_offset1;
        f.jmp_offset2 = y.jmp_offset2 + (ya as i64 - a as i64);
        f.riscv_inst = None;
        new.push((a, f));
    }
    for (a, f) in new {
        insts.get_mut(&a).unwrap().i = f;
        st.lj += 1;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: zopt <in.elf> <out.elf> [passes=jt,cf,af] [max_rounds=8]");
        std::process::exit(2);
    }
    let elf = std::fs::read(&args[1]).expect("read elf");
    let passes = args.get(3).cloned().unwrap_or("jt,cf,af".into());
    let rounds: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(8);
    let mut rom: ZiskRom = zisk_transpiler_common::program2rom(&elf).expect("transpile");
    eprintln!("insts {}", rom.insts.len());
    let (jt, cf, af, lj) = (
        passes.contains("jt"),
        passes.contains("cf"),
        passes.contains("af"),
        passes.contains("lj"),
    );
    let mut st = Stats::default();
    for round in 0..rounds {
        let before = (st.jt, st.cf, st.af, st.lj);
        if lj {
            pass_lj(&mut rom.insts, &mut st);
        }
        if cf || af {
            pass_fold(&mut rom.insts, cf, af, &mut st);
        }
        if jt {
            pass_jt(&mut rom.insts, &mut st);
        }
        eprintln!("round {round}: {st:?}");
        if before == (st.jt, st.cf, st.af, st.lj) {
            break;
        }
    }

    let out = ziskbin::rom_to_elf(&rom);
    // round-trip check
    let back = ziskbin::ziskbin2rom(&out).expect("decode").expect("ziskbin");
    assert_eq!(back.insts.len(), rom.insts.len());
    std::fs::write(&args[2], out).expect("write");
    eprintln!("wrote {}", args[2]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instruction(pc: u64, op: &str) -> ZiskInstBuilder {
        let mut builder = ZiskInstBuilder::default();
        builder.i.paddr = pc;
        builder.src_a("imm", 0, false);
        builder.src_b("imm", 0, false);
        builder.op(op).unwrap();
        builder.j(4, 4);
        builder
    }

    #[test]
    fn address_folding_preserves_operation_and_address() {
        for op in ["copyb", "signextend_b", "signextend_h", "signextend_w", "add", "eq"] {
            for load in [false, true] {
                for offset in [-8i64, 8] {
                    let mut x = instruction(100, "add");
                    x.src_a("reg", 1, false);
                    x.src_b("imm", offset as u64, false);
                    x.store("reg", 2, false, false);
                    let mut y = instruction(104, op);
                    y.src_a("reg", 2, false);
                    if load {
                        y.src_b("ind", 16, false);
                        y.store("reg", 3, false, false);
                    } else {
                        y.src_b("imm", 0x8000_0080, false);
                        y.store("ind", 16, false, false);
                    }
                    y.ind_width(8);
                    let original = y.i.clone();
                    let mut end = instruction(108, "copyb");
                    end.i.end = true;
                    let mut insts = BTreeMap::from([(100, x), (104, y), (108, end)]);
                    let mut stats = Stats::default();

                    pass_fold(&mut insts, false, true, &mut stats);

                    let should_fold = !matches!(op, "add" | "eq");
                    assert_eq!(stats.af, usize::from(should_fold), "{op}, load={load}");
                    if should_fold {
                        let folded = &insts[&100].i;
                        let base = 1000u64;
                        let original_a = base.wrapping_add(offset as u64);
                        let folded_offset =
                            if load { folded.b_offset_imm0 } else { folded.store_offset as u64 };
                        assert_eq!(folded.a_offset_imm0, 1);
                        assert_eq!(base.wrapping_add(folded_offset), original_a + 16);
                        assert_eq!(
                            ZiskOp::execute(original.op, original_a, 0x8000_0080),
                            ZiskOp::execute(folded.op, base, 0x8000_0080)
                        );
                    } else {
                        assert_eq!(insts[&100].i.op, ZiskOp::ADD);
                        assert_eq!(insts[&100].i.store_offset, 2);
                    }
                }
            }
        }
    }

    #[test]
    fn liveness_preserves_successor_cases() {
        for (j1, j2, indirect, precompiled, expected) in [
            (4, 4, false, false, 1 << 1),
            (4, 8, false, false, (1 << 1) | (1 << 2)),
            (4, 8, true, false, ALL),
            (4, 12, false, false, ALL),
            (12345, 8, false, true, 1 << 2),
        ] {
            let mut branch = instruction(100, "eq");
            branch.j(j1, j2);
            branch.i.set_pc = indirect;
            branch.i.is_precompiled = precompiled;
            let mut left = instruction(104, "copyb");
            left.src_b("reg", 1, false);
            left.i.end = true;
            let mut right = instruction(108, "copyb");
            right.src_b("reg", 2, false);
            right.i.end = true;
            let insts = BTreeMap::from([(100, branch), (104, left), (108, right)]);

            assert_eq!(liveness(&insts, &index(&insts)), vec![expected, 0, 0]);
        }
    }
}
