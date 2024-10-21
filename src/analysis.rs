use crate::opcodes::{Opcode, Operation};
use crate::utils::is_address_like;
use crate::{abstract_stack, abstract_value::*};
use alloy::primitives::U160;
use alloy::rpc::types::Block;
use itertools::Itertools;
use match_deref::match_deref;
use revm::{
    interpreter::{instructions::i256::i256_cmp, InstructionResult},
    primitives::{keccak256, Address, BlockEnv, Bytecode, FixedBytes, U256},
};

use std::borrow::Borrow;
use std::{
    borrow::BorrowMut,
    collections::{hash_map::Entry, HashMap, HashSet},
    fmt::{Debug, Display},
    hash::Hash,
    rc::Rc,
};

pub const LOGGER_TARGET_ANALYSIS: &str = "forky::analysis";

#[derive(Debug, Clone)]
pub struct AnalysisContext {
    contract_address: U256,
    gas_price: U256,
    coinbase: U256,
    block_number: U256,
    chain_id: U256,
    timestamp: U256,
    block_gas_limit: U256,
    base_fee: U256,
    prev_randao: Option<FixedBytes<32>>,
}
impl Default for AnalysisContext {
    fn default() -> Self {
        Self {
            chain_id: U256::from(1),
            contract_address: U256::ZERO,
            gas_price: U256::from(1u64),
            block_number: U256::from(0u64),
            coinbase: U256::ZERO,
            timestamp: U256::from(0u64),
            block_gas_limit: U256::from(0u64),
            base_fee: U256::from(0u64),
            prev_randao: None,
        }
    }
}
impl From<&BlockEnv> for AnalysisContext {
    fn from(env: &BlockEnv) -> Self {
        Self {
            block_number: env.number,
            coinbase: env.coinbase.into_word().into(),
            gas_price: env.basefee + U256::from(1u64),
            timestamp: env.timestamp,
            block_gas_limit: env.gas_limit,
            base_fee: env.basefee,
            prev_randao: env.prevrandao,
            ..Default::default()
        }
    }
}
impl From<&Block> for AnalysisContext {
    fn from(block: &Block) -> Self {
        Self {
            block_number: U256::from(block.header.number),
            coinbase: block.header.miner.into_word().into(),
            gas_price: U256::from(block.header.base_fee_per_gas.unwrap_or(1u64)),
            timestamp: U256::from(block.header.timestamp),
            block_gas_limit: U256::from(block.header.gas_limit),
            base_fee: U256::from(block.header.base_fee_per_gas.unwrap_or(1u64)),
            prev_randao: block.header.mix_hash,
            ..Default::default()
        }
    }
}

impl AnalysisContext {
    pub fn with_contract_address(&self, contract_address: Address) -> Self {
        Self {
            contract_address: contract_address.into_word().into(),
            ..self.clone()
        }
    }
    pub fn with_chain_id(&self, chain_id: U256) -> Self {
        Self {
            chain_id,
            ..self.clone()
        }
    }
}
/**
 * This code is a bit rough, but this code attempts to analyze contract code and try to predict
 * which addreses will be accessed during execution such that we can fetch them before running the
 * contract.
 *
 * The method used is a very basic abstract interpreter that interprets the bytecode, and tries to visit
 * every codepath once.
 *
 * Since we don't know anything about the caller nor the contract state all operators that depend on the environment
 * produce abstract values.
 *
 * When an SLOAD or SSTORE is encountered, we record the inputs and store them for later.
 *
 * When all paths are covered (or we are in an execution path that is too long), we will try and conver the
 * the abstract values into something useful. This is done in two steps:
 *  - First we recursively reduce the abstract values by looking for specific patterns, for an example:
 *    The abstract value: SLOAD(Add(SHA3(CONST(x)), CONST) => is most likely a storage slot containing an Array,
 *    located on slot x. We can replace it with StorageArray(StorageSlot(Uint, x))
 *    
 *    A more advanced example is: SLOAD(SHA3(exp, CONST(x)))
 *    Here if the 'exp' evaluates to an opcode that load from calldata or produce an address value, we know that
 *    that this storage slot will produce a mapping, and we can replace it with Mapping(?, StorageSlot(inferred type of exp, x))
 *
 *  - We will get multiple values for each storage slot, we merge this info by promoting the value to the most concrete type.
 *    For the few of you that has some knowledge of static analysis or math, it is essentially a small latice:
 *
 *    UINT < ADDRESS
 *    x < x[] < mapping(x => y) etc
 *
 */
use crate::LOGGER_TARGET_MAIN;

// static MAX_STEPS: usize = 4000;

#[warn(dead_code)]
fn sign_extend(ext: U256, x: U256) -> U256 {
    if ext < U256::from(31) {
        let ext = ext.as_limbs()[0];
        let bit_index = (8 * ext + 7) as usize;
        let bit = x.bit(bit_index);
        let mask = (U256::from(1) << bit_index) - U256::from(1);
        if bit {
            x | !mask
        } else {
            x & mask
        }
    } else {
        x
    }
}

static MAX_REVISITS: usize = 7;

pub fn decode_operation<'a>(
    bytes: &'a [u8],
    cur_offset: usize,
) -> eyre::Result<(Operation, usize)> {
    let encoded_opcode = bytes.get(cur_offset).expect("Unexpected end of input");
    let num_bytes = match *encoded_opcode {
        0x60..=0x7f => encoded_opcode - 0x5f,
        _ => 0,
    } as usize;

    let opcode = Opcode::from_byte(*encoded_opcode);
    if num_bytes > 0 {
        let input_start = cur_offset + 1;
        let input_end = input_start + num_bytes;

        if input_end > bytes.len() {
            return Err(eyre::eyre!("Invalid opcode {:?}", opcode));
        }

        Ok((
            Operation::new(opcode, cur_offset, &bytes[input_start..input_end]),
            input_end,
        ))
    } else {
        Ok((
            Operation::new(opcode, cur_offset, &bytes[0..0]),
            cur_offset + 1,
        ))
    }
}

pub fn disassemble_bytes<'a>(bytes: &'a [u8]) -> Vec<Operation<'a>> {
    let mut operations = Vec::new();
    let mut offset = 0;

    let mut len = bytes.len();
    if bytes.len() > 128 {
        for i in 1..128usize {
            let cursor = bytes.len() - i;
            let s = &bytes[cursor - 2..cursor];
            if s == [0xa2, 0x64] {
                len = cursor;
                break;
            }
        }
    }

    while offset < len {
        match decode_operation(bytes, offset) {
            Ok((operation, new_offset)) => {
                operations.push(operation);
                offset = new_offset;
            }
            Err(e) => {
                log::trace!(target: LOGGER_TARGET_MAIN, "Failed to decode opcode {:?}", e);
                break;
            }
        };
    }
    operations
}

#[derive(Debug, Clone)]
enum AbstractMemoryValue {
    Abstract {
        size: usize,
        value: AbstractValueRef,
    },
    Bytes(Vec<u8>),
}
#[derive(Debug)]
struct AbstractMemory {
    backing: HashMap<usize, AbstractMemoryValue>,
}

impl AbstractMemory {
    fn new() -> Self {
        Self {
            backing: HashMap::new(),
        }
    }

    fn load_args(
        &self,
        offset: &AbstractValueRef,
        size: &AbstractValueRef,
    ) -> (Option<AbstractValueRef>, Vec<AbstractValueRef>) {
        let (mut offset, mut size): (usize, usize) = match_deref::match_deref! {
            match (offset, size) {
                (Deref @ AbstractValue::Const(offset), Deref @ AbstractValue::Const(size)) => {
                    if size.gt(&U256::from(100000)) || offset.gt(&U256::from(100000)) {
                        return (None, Vec::new());
                    }
                    (offset.to(), size.to())
                },
                (Deref @ AbstractValue::Const(offset), _) => (offset.to(), 4),
                _ => return (None, Vec::new())
            }
        };

        let selector = self.get_word(offset).map(|v| match &*v {
            AbstractValue::Const(inner) => AbstractValue::val(&(*inner >> 224)),
            v => Rc::new(v.clone()),
        });
        if size < 4 {
            return (selector, Vec::new());
        }
        size -= 4;
        offset += 4;
        let mut args = Vec::new();
        loop {
            if let Some(arg) = self.get_word(offset) {
                args.push(arg.clone());
            } else {
                break;
            };
            if size == 0 {
                break;
            }
            offset += 32;
            if size >= 32 {
                size -= 32;
            } else {
                size = 0;
            }
        }
        (selector, args)
    }

    fn copy(&self) -> AbstractMemory {
        Self {
            backing: self.backing.clone(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.backing.keys().map(|v| *v).max().unwrap_or(0usize)
    }

    #[inline]
    fn load_bytes_const(&self, offset: usize, read_size: usize) -> AbstractMemoryValue {
        let data = self.backing.get(&offset);
        if let Some(res) = data {
            match res {
                AbstractMemoryValue::Abstract { size, value } => {
                    if *size == read_size {
                        return AbstractMemoryValue::Abstract {
                            size: size.clone(),
                            value: value.clone(),
                        };
                    }
                }
                AbstractMemoryValue::Bytes(v) => return AbstractMemoryValue::Bytes(v.clone()),
            }
        }
        return AbstractMemoryValue::Abstract {
            size: read_size,
            value: AbstractValue::bin(
                Opcode::MLOAD,
                HeapRef::new(AbstractValue::Const(U256::from(offset))),
                HeapRef::new(AbstractValue::Const(U256::from(read_size))),
            ),
        };
    }

    #[inline]
    fn store(&mut self, offset: usize, value: AbstractMemoryValue) {
        self.backing.insert(offset, value);
    }

    #[inline]
    fn mstore(&mut self, offset: AbstractValueRef, value: AbstractValueRef) {
        let offset: usize = match_deref::match_deref! {
            match &offset {
                Deref @ AbstractValue::Const(v) => v.to(),
                _ => return (),
            }
        };
        self.backing.insert(
            offset,
            AbstractMemoryValue::Abstract {
                size: 32,
                value: value.clone(),
            },
        );
    }

    #[inline]
    fn mstore8(&mut self, offset: AbstractValueRef, value: AbstractValueRef) {
        let offset: usize = match_deref::match_deref! {
            match &offset {
                    Deref @  AbstractValue::Const(v) => v.to(),
                _ => return (),
            }
        };
        self.backing.insert(
            offset,
            AbstractMemoryValue::Abstract {
                size: 1,
                value: value.clone(),
            },
        );
    }
    #[inline]
    fn get_word(&self, offset: usize) -> Option<AbstractValueRef> {
        return match self.backing.get(&offset) {
            Some(v) => Some(match v {
                AbstractMemoryValue::Abstract { size, value } => value.clone(),
                AbstractMemoryValue::Bytes(v) => {
                    if v.len() <= 32 {
                        let val = U256::from_be_slice(v.as_slice());
                        // println!("get_word({:?}) = {:?}", offset, val);
                        return Some(HeapRef::new(AbstractValue::Const(val)));
                    }
                    // println!("get_word({:?}) = {:?}", offset, v);
                    return Some(HeapRef::new(AbstractValue::UnaryOpResult(
                        Opcode::MLOAD,
                        HeapRef::new(AbstractValue::Const(U256::from(offset))),
                    )));
                }
            }),
            None => None,
        };
    }
    #[inline]
    fn load(&self, offset_v: AbstractValueRef) -> Option<AbstractValueRef> {
        let offset = match_deref::match_deref! {
            match &offset_v {
                Deref @ AbstractValue::Const(v) => {
                    if v.gt(&U256::from(10000)) {
                        return Some(HeapRef::new(AbstractValue::UnaryOpResult(Opcode::MLOAD, offset_v)))
                    }
                    v.clone().to()
                },
                _ => return Some(HeapRef::new(AbstractValue::UnaryOpResult(Opcode::MLOAD, offset_v)))
            }
        };
        self.get_word(offset)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum AnalyzedStoragesSlot {
    Slot(U256, SlotType),
    Array(U256, SlotType),
    Mapping(U256, Vec<SlotType>, SlotType, Vec<Vec<ArgType>>),
}

impl AnalyzedStoragesSlot {
    pub fn is_mapping(&self) -> bool {
        match self {
            AnalyzedStoragesSlot::Mapping(_, _, _, _) => true,
            _ => false,
        }
    }
    fn promote(&self, other: &AnalyzedStoragesSlot) -> Self {
        match (self, other) {
            (AnalyzedStoragesSlot::Slot(slot, typ), AnalyzedStoragesSlot::Array(_, other_typ)) => {
                AnalyzedStoragesSlot::Array(*slot, typ.promote(other_typ))
            }
            (
                AnalyzedStoragesSlot::Slot(slot, typ),
                AnalyzedStoragesSlot::Mapping(_, other_map, other_typ, args),
            ) => AnalyzedStoragesSlot::Mapping(
                *slot,
                other_map.clone(),
                typ.promote(other_typ),
                args.clone(),
            ),
            (
                AnalyzedStoragesSlot::Array(slot, typ),
                AnalyzedStoragesSlot::Mapping(_, other_map, other_typ, args),
            ) => AnalyzedStoragesSlot::Mapping(
                *slot,
                other_map.clone(),
                typ.promote(other_typ),
                args.clone(),
            ),
            (
                AnalyzedStoragesSlot::Mapping(_, map0, t0, arg0),
                AnalyzedStoragesSlot::Mapping(slot, map1, t1, arg1),
            ) => {
                let mapping = if map0.len() > map1.len() {
                    map0.clone()
                } else if map1.len() > map0.len() {
                    map1.clone()
                } else {
                    map0.iter()
                        .zip(map1.iter())
                        .map(|(a, b)| a.promote(b))
                        .collect::<Vec<_>>()
                };

                let args = arg0
                    .iter()
                    .chain(arg1.iter())
                    .cloned()
                    .filter(|v| v.len() == mapping.len())
                    .unique()
                    .collect::<Vec<_>>();

                AnalyzedStoragesSlot::Mapping(*slot, mapping, t0.promote(t1), args)
            }
            _ => self.promote_type(&other.get_type()),
        }
    }
    fn promote_type(&self, new_type: &SlotType) -> AnalyzedStoragesSlot {
        match self {
            AnalyzedStoragesSlot::Slot(slot, typ) => {
                AnalyzedStoragesSlot::Slot(*slot, typ.promote(new_type))
            }
            AnalyzedStoragesSlot::Array(slot, typ) => {
                if let SlotType::Tuple(_) = new_type {
                    return self.clone();
                }
                if new_type == &SlotType::String {
                    AnalyzedStoragesSlot::Slot(*slot, SlotType::String)
                } else {
                    AnalyzedStoragesSlot::Array(*slot, typ.promote(new_type))
                }
            }
            AnalyzedStoragesSlot::Mapping(slot, map, typ, args) => AnalyzedStoragesSlot::Mapping(
                *slot,
                map.clone(),
                typ.promote(new_type),
                args.clone(),
            ),
        }
    }

    pub fn get_slot(&self) -> U256 {
        match self {
            AnalyzedStoragesSlot::Slot(slot, _) => *slot,
            AnalyzedStoragesSlot::Array(slot, _) => *slot,
            AnalyzedStoragesSlot::Mapping(slot, __, _, _) => *slot,
        }
    }

    fn get_mapping(&self) -> Vec<SlotType> {
        match self {
            AnalyzedStoragesSlot::Mapping(_, map, _, _) => map.clone(),
            _ => vec![],
        }
    }

    pub fn get_type(&self) -> SlotType {
        match self {
            AnalyzedStoragesSlot::Slot(_, typ) => typ.clone(),
            AnalyzedStoragesSlot::Array(_, typ) => typ.clone(),
            AnalyzedStoragesSlot::Mapping(_, _, typ, _) => typ.clone(),
        }
    }
}

impl Display for AnalyzedStoragesSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalyzedStoragesSlot::Slot(slot, typ) => write!(f, "Slot({slot}, {typ:?})"),
            AnalyzedStoragesSlot::Array(slot, typ) => {
                let start: U256 = keccak256(slot.to_be_bytes::<32>().as_slice()).into();
                write!(f, "Slot({slot}, {typ:?}[]) array_start={}", start)
            }
            AnalyzedStoragesSlot::Mapping(slot, map, typ, args) => write!(
                f,
                "Slot({slot}, mapping({} => {}), args={:?})",
                map.iter().join(" => "),
                typ,
                args
            ),
        }
    }
}

fn convert(
    v: &AbstractValueRef,
    arg_source: &Vec<ArgType>,
    selector: u32,
) -> Option<AnalyzedStoragesSlot> {
    use crate::abstract_value::AbstractValue::*;
    let out = match_deref! {
        match v {
            Deref @ StorageSlot(typ, slot) => Some(AnalyzedStoragesSlot::Slot(*slot, typ.clone())),
            Deref @ TypeMask(t, inner) => convert(inner, arg_source, selector).map(|v| v.promote_type(&t)),
            Deref @ StorageArray(slot) => {
                let slot = convert(slot, &Vec::new(), selector)?;
                Some(AnalyzedStoragesSlot::Array(slot.get_slot(), slot.get_type()))
            }
            Deref @ Mapping(arg_type, slot) => {
                let arg_source = arg_source.iter().map(|i|i.clone()).unique().collect::<Vec<_>>();
                let slot = convert(slot, &arg_source, selector)?;
                let mut mapping = vec![arg_type.clone()];
                mapping.extend(slot.get_mapping());
                Some(
                AnalyzedStoragesSlot::Mapping(slot.get_slot(), mapping, slot.get_type(), vec![
                    arg_source
                ]))
            }
            _ => {
                return None
            }
        }
    };

    out
}

macro_rules! handle_unary {
    ($stack:expr, $op:ident, $f:expr) => {{
        let val = $stack.pop()?;

        match_deref::match_deref! {
            match &val {
                Deref @ AbstractValue::Const(v) => {
                    $stack.push(AbstractValue::val(&U256::from($f(v))))?;
                }
                _ => {
                    $stack.push(AbstractValue::unary(Opcode::$op, val.clone()))?;
                }
            }
        };
    }};
}

macro_rules! handle_bin {
    ($stack:expr, $op:tt, $opName:ident) => {{
        let a = $stack.pop()?;
        let b = $stack.pop()?;
        match_deref::match_deref! {
            match (&a, &b) {
                (Deref@AbstractValue::Const(a), Deref@AbstractValue::Const(b)) => {
                    let res = U256::from(a.clone() $op b.clone());

                    $stack.push(HeapRef::new(AbstractValue::Const(res)))?;
                }
                _ => {
                    $stack.push(AbstractValue::bin(Opcode::$opName, a.clone(), b.clone()))?;
                }
            }
        };
    }};
    ($stack:expr, $op:expr, $opName:ident) => {{
        let a = $stack.pop()?;
        let b = $stack.pop()?;
        match_deref::match_deref! {
            match (&a, &b) {
                (Deref@AbstractValue::Const(a), Deref@AbstractValue::Const(b)) => {
                    let res = U256::from($op(a.clone(), b.clone()));
                    // println!("{} {:?} {:?} -> {:?}", stringify!($opName), a, b, res);
                    $stack.push(HeapRef::new(AbstractValue::Const(res)))?;
                }
                _ => {
                    $stack.push(AbstractValue::bin(Opcode::$opName, a.clone(), b.clone()))?;
                }
            }
        };
    }};
}

macro_rules! handle_shift_op {
    ($stack:expr, $op:expr, $opName:ident, $max:expr) => {{
        let shift = $stack.pop()?;
        let value = $stack.pop()?;
        match_deref::match_deref! {
            match (&shift, &value) {
                (Deref @ AbstractValue::Const(shift), Deref @ AbstractValue::Const(value)) => {
                    if shift.gt(&U256::from(255)) {
                        $stack.push(HeapRef::new(AbstractValue::Const($max)))?;
                    } else {
                        let shift: usize = shift.to();
                        let res = U256::from($op(*value, shift));
                        $stack.push(HeapRef::new(AbstractValue::Const(res)))?;
                    }
                }

                (Deref @ AbstractValue::Const(shift_value), _) => {
                    if shift_value.gt(&U256::from(255)) {
                        $stack.push(HeapRef::new(AbstractValue::Const($max)))?;
                    } else {
                        $stack.push(AbstractValue::bin(Opcode::$opName, shift.clone(), value.clone()))?;
                    }
                }
                _ => {
                    $stack.push(AbstractValue::bin(Opcode::$opName, shift.clone(), value.clone()))?;
                }
            }
        };


    }};
}

macro_rules! handle_bin_c {
    ($stack:expr, $op:expr, $opName:ident) => {{
        let a = $stack.pop()?;
        let b = $stack.pop()?;

        match_deref::match_deref! {
            match (&a, &b) {
                (Deref @ AbstractValue::Const(a), Deref @ AbstractValue::Const(b)) => {
                    let res = if $op(&a.clone(), &b.clone()) {
                        U256::from(1u64)
                    } else {
                        U256::ZERO
                    };
                    $stack.push(HeapRef::new(AbstractValue::Const(res)))?;
                }
                (a, b) => {

                    $stack.push(AbstractValue::bin(Opcode::$opName, a.clone(), b.clone()))?;
                }
            }
        };
    }};
}

enum StepResult {
    Ok,
    Stop,
    JUMP,
    Split(usize),
}
struct AbstractVMInstance<'a> {
    program_bytes: &'a [u8],
    program: &'a Vec<Operation<'a>>,
    pc: usize,
    jump_dests: Rc<HashMap<usize, usize>>,
    stack: abstract_stack::AbstractStack,
    memory: AbstractMemory,
    storage: HashMap<U256, AbstractValueRef>,
    tmemory: HashMap<U256, AbstractValueRef>,
    steps: usize,

    last_push4: u32,
    last_selector: u32,
    last_selector_cmd: usize,
    return_data_size: usize,

    halted: bool,
}
impl<'a> AbstractVMInstance<'a> {
    fn copy(&self, pc: usize) -> Self {
        Self {
            program_bytes: self.program_bytes,
            program: self.program,
            jump_dests: self.jump_dests.clone(),
            pc,
            stack: self.stack.copy(),
            memory: self.memory.copy(),
            storage: self.storage.clone(),
            tmemory: self.tmemory.clone(),
            halted: self.halted,
            last_selector: self.last_selector,
            last_push4: self.last_push4,
            return_data_size: self.return_data_size,
            steps: self.steps,
            last_selector_cmd: self.last_selector_cmd,
        }
    }
    fn new(
        program_bytes: &'a [u8],
        program: &'a Vec<Operation>,
        jump_dests: Rc<HashMap<usize, usize>>,
        pc: usize,
    ) -> Self {
        Self {
            program_bytes: program_bytes,
            program,
            pc,
            jump_dests,
            stack: abstract_stack::AbstractStack::new(),
            memory: AbstractMemory::new(),
            storage: HashMap::new(),
            tmemory: HashMap::new(),
            steps: 0,
            last_push4: 0,
            last_selector: 0,
            return_data_size: 0,
            halted: false,
            last_selector_cmd: 0,
        }
    }

    pub fn step(
        &mut self,
        analysis: &mut Analysis,
        context: &AnalysisContext,
    ) -> Result<StepResult, InstructionResult> {
        if self.halted {
            return Ok(StepResult::Stop);
        }

        // if self.steps > MAX_STEPS {
        //     // println!("MAX STEPS");
        //     self.halted = true;
        //     return Ok(StepResult::Stop);
        // }

        let ins = self.program.get(self.pc);

        // println!("{:?}", ins);
        let res = match ins {
            None => return Ok(StepResult::Stop),
            Some(v) => self.step_(v, analysis, context)?,
        };
        let top = self.stack.data.last();
        if let Some(val) = top {
            if let AbstractValue::Const(value) = &**val {
                if is_address_like(value) {
                    analysis
                        .external_contracts
                        .push(HeapRef::new(AbstractValue::AddressConst(Address::from(
                            value.to::<U160>(),
                        ))));
                }
            }
        }
        match &res {
            StepResult::Stop => {
                self.halted = true;
            }
            _ => {}
        }
        self.steps += 1;
        Ok(res)
    }

    fn step_(
        &mut self,
        ins: &Operation,
        analysis: &mut Analysis,
        context: &AnalysisContext,
    ) -> Result<StepResult, InstructionResult> {
        let stack = &mut self.stack;
        let memory = &mut self.memory;
        let tmemory = &mut self.tmemory;
        let jump_dests = &self.jump_dests;

        match ins.opcode {
            Opcode::SELFDESTRUCT => {
                stack.drop_n(1)?;
                return Ok(StepResult::Stop);
            }
            Opcode::STOP | Opcode::INVALID => {
                return Ok(StepResult::Stop);
            }
            Opcode::REVERT | Opcode::RETURN => {
                stack.drop_n(2)?;
                return Ok(StepResult::Stop);
            }
            Opcode::JUMP => {
                let byte_offset = stack.pop()?;

                match_deref::match_deref! {
                    match &byte_offset {
                        Deref @ AbstractValue::Const(v) => {
                            let offset: usize = v.to();
                            if let Some(new_pc) = jump_dests.get(&offset) {
                                self.pc = *new_pc;
                                return Ok(StepResult::JUMP);
                            }
                        }
                        _ => {}
                    }
                };

                return Ok(StepResult::Stop);
            }
            Opcode::JUMPI => {
                let byte_offset = stack.pop()?;
                let cond = stack.pop()?;
                match_deref::match_deref! {
                    match (&cond, &byte_offset) {
                        (Deref @ AbstractValue::Const(cond), Deref @ AbstractValue::Const(byte_offset)) => {
                            let offset: usize = byte_offset.to();
                            if let Some(new_pc) = jump_dests.get(&offset) {
                                if cond.is_zero() {
                                    self.pc += 1;
                                } else {
                                    self.pc = *new_pc;
                                }
                                return Ok(StepResult::Ok);
                            }
                        }
                        _ => {}
                    }
                };

                match_deref! {
                    match &cond {
                        Deref @ AbstractValue::TypeMask(SlotType::Bool, inner) => {
                            if let Some(slot) = inner.slot() {
                                analysis.promote(&slot, SlotType::Bool);
                            };
                        }
                        _ => ()
                    }
                }

                self.pc += 1;
                match_deref::match_deref! {
                    match &byte_offset {
                        Deref @ AbstractValue::Const(byte_offset) => {
                            let offset: usize = byte_offset.to();
                            if let Some(new_pc) = jump_dests.get(&offset) {
                                return Ok(StepResult::Split(*new_pc));
                            }
                        }
                        _ => {
                        }
                    }
                };

                return Ok(StepResult::Ok);
            }
            _ => {}
        };
        match ins.opcode {
            Opcode::STATICCALL | Opcode::DELEGATECALL => {
                let gas = stack.pop()?;
                let address = stack.pop()?;
                analysis.external_contracts.push(address.clone());

                let args_offset = stack.pop()?;
                let args_size = stack.pop()?;
                let ret_offset = stack.pop()?;
                let ret_size = stack.pop()?;
                if let AbstractValue::Const(v) = *ret_size {
                    let size: usize = v.to();
                    self.return_data_size = size;
                }
                let (selector, args) = memory.load_args(&args_offset, &args_size);

                let res = HeapRef::new(AbstractValue::static_or_delegate_call(
                    ins.opcode, gas, address, selector, args,
                ));

                memory.mstore(ret_offset, res.clone());

                stack.push(res.clone())?;
            }
            Opcode::CALL | Opcode::CALLCODE => {
                let gas = stack.pop()?;
                let address = stack.pop()?;

                analysis.external_contracts.push(address.clone());

                let value = stack.pop()?;
                let args_offset = stack.pop()?;
                let args_size = stack.pop()?;
                let ret_offset = stack.pop()?;
                let ret_size = stack.pop()?;
                if let AbstractValue::Const(v) = *ret_size {
                    let size: usize = v.to();
                    self.return_data_size = size;
                }

                let (selector, args) = memory.load_args(&args_offset, &args_size);

                let res = HeapRef::new(AbstractValue::ext_call(
                    ins.opcode, gas, address, value, selector, args,
                ));
                memory.mstore(ret_offset, res.clone());
                stack.push(res)?;
            }

            Opcode::CREATE => {
                stack.drop_n(3)?;
                stack.push(AbstractValue::op(Opcode::CREATE))?
            }

            Opcode::CREATE2 => {
                stack.drop_n(4)?;
                stack.push(AbstractValue::op(Opcode::CREATE2))?
            }

            Opcode::ADD => handle_bin!(stack, U256::wrapping_add, ADD),
            Opcode::SUB => handle_bin!(stack, U256::wrapping_sub, SUB),
            Opcode::MUL => handle_bin!(stack, U256::wrapping_mul, MUL),
            Opcode::DIV => {
                let value = match_deref::match_deref! {
                    match (&stack.pop()?, &stack.pop()?) {
                        (Deref @ AbstractValue::Const(a), Deref @ AbstractValue::Const(b)) => {
                            if b.is_zero() {
                                AbstractValue::val(&a)
                            } else {
                                AbstractValue::val(&a.wrapping_div(*b))
                            }
                        },
                        (a, b) => AbstractValue::bin(Opcode::DIV, a.clone(), b.clone())
                    }
                };
                stack.push(value)?;
            }
            Opcode::AND => {
                handle_bin!(stack, &, AND);
                match_deref! {
                    match stack.data.last() {
                        Some(Deref@AbstractValue::TypeMask(t, any)) => {
                            if t == &SlotType::Address {
                                analysis.external_contracts.push(HeapRef::new(AbstractValue::TypeMask(*t, any.clone())));
                            }
                            if let Some(slot) = any.slot() {
                                analysis.promote(&slot, *t);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Opcode::SHL => handle_shift_op!(stack, U256::wrapping_shl, SHL, U256::ZERO),
            Opcode::SHR => handle_shift_op!(stack, U256::wrapping_shr, SHR, U256::MAX),
            Opcode::MOD => handle_bin!(stack, U256::wrapping_rem, MOD),
            Opcode::SDIV => {
                handle_bin!(stack, revm::interpreter::instructions::i256::i256_div, SDIV)
            }
            Opcode::EQ => {
                if self.last_push4 != 0
                    && self.pc > self.last_selector_cmd
                    && self.pc - self.last_selector_cmd < 8
                {
                    self.last_selector = self.last_push4;
                    self.last_push4 = 0;
                }
                handle_bin_c!(stack, U256::eq, EQ);
            }
            Opcode::LT => handle_bin_c!(stack, U256::lt, LT),
            Opcode::GT => handle_bin_c!(stack, U256::gt, GT),
            Opcode::NOT => {
                let v = stack.pop()?;
                stack.push(match_deref! {
                    match &v {
                        Deref @ AbstractValue::Const(v) => AbstractValue::val(&(!v)),
                        _ => AbstractValue::unary(Opcode::NOT, v)
                    }
                })?;
            }
            Opcode::ISZERO => handle_unary!(stack, ISZERO, U256::is_zero),
            Opcode::POP => stack.drop_n(1)?,
            Opcode::OR => handle_bin!(stack, |, OR),
            Opcode::XOR => handle_bin!(stack, ^, XOR),
            Opcode::SMOD => {
                handle_bin!(stack, revm::interpreter::instructions::i256::i256_mod, SMOD)
            }

            Opcode::MULMOD | Opcode::ADDMOD => {
                let a = stack.pop()?;
                let b = stack.pop()?;
                let c = stack.pop()?;
                match_deref::match_deref! {
                    match (&a, &b, &c) {
                        (
                            Deref @ AbstractValue::Const(a),
                            Deref @ AbstractValue::Const(b),
                            Deref @ AbstractValue::Const(c),
                        ) => {
                            if ins.opcode == Opcode::MULMOD {
                                stack.push(
                                    HeapRef::new(AbstractValue::Const(U256::mul_mod(a.clone(), b.clone(), c.clone())))
                                )?;
                            } else {
                                stack.push(
                                    HeapRef::new(AbstractValue::Const(U256::add_mod(a.clone(), b.clone(), c.clone())))
                                )?;
                            }
                        }
                        _ => {
                            stack.push(
                                HeapRef::new(AbstractValue::TertiaryOpResult(ins.opcode, a, b, c))
                            )?;
                        }
                    }
                }
            }
            Opcode::EXP => handle_bin!(stack, U256::pow, EXP),
            Opcode::SIGNEXTEND => handle_bin!(stack, crate::analysis::sign_extend, SIGNEXTEND),
            Opcode::SLT | Opcode::SGT => {
                let order = if ins.opcode == Opcode::SLT {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
                let a = stack.pop()?;
                let b = stack.pop()?;
                match_deref::match_deref! {
                    match (&a, &b) {
                        (Deref @ AbstractValue::Const(a), Deref @ AbstractValue::Const(b)) => {
                            stack.push(
                                HeapRef::new(AbstractValue::Const(U256::from(
                                    i256_cmp(a, b) == order
                                )))
                            )?;
                        }
                        _ => {
                            stack.push(
                                AbstractValue::bin(ins.opcode, a, b)
                            )?;
                        }
                    }
                }
            }
            Opcode::BYTE => {
                let i = stack.pop()?;
                let w = stack.pop()?;
                match_deref::match_deref! {
                    match (&w, &i) {
                        (Deref @ AbstractValue::Const(w), Deref @ AbstractValue::Const(i)) => {
                            let i: usize = i.to();
                            let b = w.byte(31 - i);
                            stack.push(HeapRef::new(AbstractValue::Const(U256::from(b))))?;
                        }
                        _ => {
                            stack.push(
                                AbstractValue::bin(Opcode::BYTE, w, i)
                            )?;
                        }
                    }
                }
            }
            Opcode::SAR => handle_shift_op!(stack, U256::arithmetic_shr, SAR, U256::MAX),
            Opcode::SHA3 => {
                let offset = stack.pop()?;
                let size = stack.pop()?;

                let exact = match_deref::match_deref! {
                    match (&offset, &size) {
                        (Deref @ AbstractValue::Const(offset), Deref @ AbstractValue::Const(size)) => {
                            if offset.gt(&U256::from(10000)) || size.gt(&U256::from(96)) {
                                None
                            } else {
                                let offset: usize = offset.to();
                                let size: usize = size.to();
                                let out = if size == 32 {
                                    memory.get_word(offset).map(|v|AbstractValue::unary(Opcode::SHA3, v))
                                } else if size == 64 {
                                    if let (Some(v0), Some(v1)) = (memory.get_word(offset), memory.get_word(offset+32)) {
                                        Some(AbstractValue::bin(Opcode::SHA3, v0, v1))
                                    } else {
                                        None
                                    }
                                } else if size == 96 {
                                    if let (Some(v0), Some(v1), Some(v2)) = (memory.get_word(offset), memory.get_word(offset+32), memory.get_word(offset+64)) {
                                        Some(HeapRef::new(AbstractValue::TertiaryOpResult(Opcode::SHA3, v0, v1, v2)))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };
                                out
                            }
                        }
                        _ => {
                            None
                        }
                    }
                };

                if let Some(v) = exact {
                    stack.push(v)?;
                } else {
                    let value = memory.load(offset.clone());
                    match value {
                        Some(v) => {
                            stack.push(AbstractValue::unary(Opcode::SHA3, v))?;
                        }
                        None => {
                            stack.push(AbstractValue::bin(Opcode::SHA3, offset, size))?;
                        }
                    }
                }
            }
            Opcode::CALLDATALOAD => {
                let offset = stack.pop()?;
                match_deref::match_deref! {
                    match &offset {
                        Deref @ AbstractValue::Const(v) => {
                            if v.lt(&U256::from(10000)) {
                                let offset: usize = v.to();
                                let arg_index = if offset > 4 {
                                    (offset - 4) / 32
                                } else {
                                    0
                                };
                                stack.push(HeapRef::new(AbstractValue::Calldata(arg_index)))?;
                            } else {
                                stack.push(AbstractValue::unary(Opcode::CALLDATALOAD, offset))?
                            }
                        }
                        _ => {
                            stack.push(AbstractValue::unary(Opcode::CALLDATALOAD, offset))?
                        }
                    }
                };
            }
            Opcode::CODESIZE => stack.push(AbstractValue::val(&U256::from(self.program.len())))?,
            Opcode::BALANCE | Opcode::EXTCODESIZE | Opcode::EXTCODEHASH => {
                let address = stack.pop()?;
                analysis.external_contracts.push(address.clone());
                stack.push(AbstractValue::op(ins.opcode))?
            }
            Opcode::EXTCODECOPY => {
                let address = stack.pop()?;
                analysis.external_contracts.push(address.clone());
                stack.pop()?;
                let offset = stack.pop()?;
                stack.pop()?;
                memory.mstore(offset, AbstractValue::op(Opcode::EXTCODECOPY));
            }
            Opcode::RETURNDATACOPY | Opcode::CALLDATACOPY | Opcode::CODECOPY => {
                stack.drop_n(1)?;
                let offset = stack.pop()?;
                stack.drop_n(1)?;
                memory.mstore(offset, AbstractValue::op(ins.opcode));
            }
            Opcode::BLOBHASH => {
                stack.drop_n(1)?;
                stack.push(AbstractValue::op(Opcode::BLOBHASH))?
            }
            Opcode::BLOCKHASH => {
                let val = stack.pop()?;
                stack.push(AbstractValue::unary(Opcode::BLOCKHASH, val))?;
            }

            Opcode::RETURNDATASIZE => stack.push_uint(self.return_data_size as u64)?,

            // All of these could technically be context dependent..
            Opcode::ORIGIN
            | Opcode::GAS
            | Opcode::CALLVALUE
            | Opcode::CALLER
            | Opcode::ADDRESS
            | Opcode::CALLDATASIZE
            | Opcode::BLOBBASEFEE
            | Opcode::DIFFICULTY
            | Opcode::SELFBALANCE => stack.push(AbstractValue::op(ins.opcode))?,
            Opcode::TIMESTAMP => stack.push(AbstractValue::val(&context.timestamp))?,
            Opcode::NUMBER => stack.push(AbstractValue::val(&context.block_number))?,
            Opcode::COINBASE => stack.push(AbstractValue::val(&context.coinbase))?,
            Opcode::CHAINID => stack.push(AbstractValue::val(&context.chain_id))?,
            Opcode::BASEFEE => stack.push(AbstractValue::val(&context.base_fee))?,
            Opcode::GASPRICE => stack.push(AbstractValue::val(&context.gas_price))?,
            // Opcode::ADDRESS => stack.push(AbstractValue::val(&context.contract_address))?,
            Opcode::GASLIMIT => stack.push(AbstractValue::val(&context.block_gas_limit))?,

            Opcode::MLOAD => {
                let offset = &stack.pop()?;
                match memory.load(offset.clone()) {
                    Some(v) => stack.push(v)?,
                    None => stack.push(AbstractValue::unary(Opcode::MLOAD, offset.clone()))?,
                };
            }
            Opcode::MSTORE => memory.mstore(stack.pop()?, stack.pop()?),
            Opcode::MSTORE8 => memory.mstore8(stack.pop()?, stack.pop()?),
            Opcode::SLOAD => {
                let sload_offset = AbstractValue::unary(Opcode::SLOAD, stack.pop()?);
                let args = sload_offset.arg_source(self.last_selector);
                // println!("SLOAD: {:?}", sload_offset);
                let reduced = reduce(&sload_offset, &args);
                stack.push(reduced.clone())?;

                if let Some(slot) = convert(&reduced, &args, self.last_selector) {
                    analysis.storage_slots.push(slot);
                }
            }
            Opcode::SSTORE => {
                let sstore_offset = AbstractValue::unary(Opcode::SLOAD, stack.pop()?);
                let args = sstore_offset.arg_source(self.last_selector);
                let reduced = reduce(&sstore_offset, &args);
                stack.pop()?;

                if let Some(slot) = convert(&reduced, &args, self.last_selector) {
                    analysis.storage_slots.push(slot);
                }
            }
            Opcode::PC => stack.push(AbstractValue::val(&U256::from(ins.offset)))?,
            Opcode::MSIZE => stack.push(AbstractValue::val(&U256::from(memory.len())))?,
            Opcode::MCOPY => {
                match_deref::match_deref! {
                    match (&stack.pop()?, &stack.pop()?, &stack.pop()?) {
                        (
                            Deref @ AbstractValue::Const(dest),
                            Deref @ AbstractValue::Const(offset),
                            Deref @ AbstractValue::Const(size),
                        ) => {
                            let dest: usize = dest.to();
                            let offset: usize = offset.to();
                            let size: usize = size.to();
                            let value =
                                memory.load_bytes_const(offset, size);

                            memory.store(dest, value);
                        }
                        _ => {}
                    }
                }
            }
            Opcode::TLOAD => {
                let offset_v = stack.pop()?;

                let res = match_deref::match_deref! {
                    match &offset_v {
                        Deref @ AbstractValue::Const(offset) => tmemory
                            .get(&offset)
                            .unwrap_or(&HeapRef::new(AbstractValue::UnaryOpResult(
                                Opcode::TLOAD,
                                offset_v
                            )))
                            .clone(),
                        _ => HeapRef::new(AbstractValue::UnaryOpResult(
                            Opcode::TLOAD,
                            offset_v
                        )),
                    }
                };
                stack.push(res)?;
            }
            Opcode::TSTORE => {
                let offset = stack.pop()?;
                let value = stack.pop()?;
                match_deref::match_deref! {
                    match &offset {
                        Deref @ AbstractValue::Const(offset) => {
                            tmemory.insert(offset.clone(), value.clone());
                        }
                        _ => {}
                    }
                };
            }
            Opcode::DUP1 => stack.dup(1)?,
            Opcode::DUP2 => stack.dup(2)?,
            Opcode::DUP3 => stack.dup(3)?,
            Opcode::DUP4 => stack.dup(4)?,
            Opcode::DUP5 => stack.dup(5)?,
            Opcode::DUP6 => stack.dup(6)?,
            Opcode::DUP7 => stack.dup(7)?,
            Opcode::DUP8 => stack.dup(8)?,
            Opcode::DUP9 => stack.dup(9)?,
            Opcode::DUP10 => stack.dup(10)?,
            Opcode::DUP11 => stack.dup(11)?,
            Opcode::DUP12 => stack.dup(12)?,
            Opcode::DUP13 => stack.dup(13)?,
            Opcode::DUP14 => stack.dup(14)?,
            Opcode::DUP15 => stack.dup(15)?,
            Opcode::DUP16 => stack.dup(16)?,
            Opcode::LOG0 => stack.drop_n(2)?,
            Opcode::LOG1 => stack.drop_n(3)?,
            Opcode::LOG2 => stack.drop_n(4)?,
            Opcode::LOG3 => stack.drop_n(5)?,
            Opcode::LOG4 => stack.drop_n(6)?,
            Opcode::SWAP1 => stack.swap(1)?,
            Opcode::SWAP2 => stack.swap(2)?,
            Opcode::SWAP3 => stack.swap(3)?,
            Opcode::SWAP4 => stack.swap(4)?,
            Opcode::SWAP5 => stack.swap(5)?,
            Opcode::SWAP6 => stack.swap(6)?,
            Opcode::SWAP7 => stack.swap(7)?,
            Opcode::SWAP8 => stack.swap(8)?,
            Opcode::SWAP9 => stack.swap(9)?,
            Opcode::SWAP10 => stack.swap(10)?,
            Opcode::SWAP11 => stack.swap(11)?,
            Opcode::SWAP12 => stack.swap(12)?,
            Opcode::SWAP13 => stack.swap(13)?,
            Opcode::SWAP14 => stack.swap(14)?,
            Opcode::SWAP15 => stack.swap(15)?,
            Opcode::SWAP16 => stack.swap(16)?,
            Opcode::PUSH0 => stack.push(AbstractValue::val(&U256::ZERO))?,
            Opcode::PUSH1 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH2 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH3 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH4 => {
                self.last_push4 =
                    u32::from_be_bytes([ins.input[3], ins.input[2], ins.input[1], ins.input[0]]);
                self.last_selector_cmd = self.pc;
                stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?
            }
            Opcode::PUSH5 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH6 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH7 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH8 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH9 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH10 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH11 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH12 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH13 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH14 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH15 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH16 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH17 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH18 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH19 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH20 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH21 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH22 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH23 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH24 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH25 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH26 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH27 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH28 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH29 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH30 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH31 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::PUSH32 => stack.push(AbstractValue::val(&U256::from_be_slice(ins.input)))?,
            Opcode::JUMPDEST => {}
            op => {
                panic!("Abstract opcode not implemented {:?}", op);
            }
        };
        self.pc += 1;

        return Ok(StepResult::Ok);
    }
}

struct Analysis {
    storage_slots: Vec<AnalyzedStoragesSlot>,
    external_contracts: Vec<AbstractValueRef>,
    slot_mask_size: HashMap<U256, SlotType>,
}

impl Analysis {
    fn new() -> Self {
        Self {
            storage_slots: Vec::new(),
            external_contracts: Vec::new(),
            slot_mask_size: HashMap::new(),
        }
    }

    fn promote(&mut self, slot: &U256, typ: SlotType) {
        match self.slot_mask_size.entry(*slot) {
            Entry::Occupied(mut entry) => {
                entry.insert(entry.get().promote(&typ));
            }
            Entry::Vacant(entry) => {
                entry.insert(typ);
            }
        }
    }
}
struct AbstractVM<'a> {
    analysis: Analysis,
    stack: Vec<AbstractVMInstance<'a>>,
    jump_dests: Rc<HashMap<usize, usize>>,
}

impl<'a> AbstractVM<'a> {
    fn new(
        program_bytes: &'a [u8],
        program: &'a Vec<Operation>,
        jump_dests: Rc<HashMap<usize, usize>>,
        start: usize,
    ) -> Self {
        let current = AbstractVMInstance::new(program_bytes, program, jump_dests.clone(), start);

        Self {
            analysis: Analysis::new(),
            stack: vec![current],
            jump_dests,
        }
    }

    pub fn run(&mut self, context: &AnalysisContext) {
        let mut splits: HashMap<usize, usize> = HashMap::new();

        loop {
            let mut vm = match self.stack.pop() {
                None => {
                    break;
                }
                Some(v) => {
                    if v.halted {
                        continue;
                    }
                    v
                }
            };

            loop {
                let res = match vm.step(self.analysis.borrow_mut(), context) {
                    Ok(v) => v,
                    Err(e) => {
                        log::debug!(target: LOGGER_TARGET_MAIN, "Failed to analyze program: {:?}", e);
                        break;
                    }
                };

                match res {
                    StepResult::Stop => break,
                    StepResult::JUMP => {
                        let current = splits.entry(vm.pc).or_insert(1);
                        if *current >= MAX_REVISITS {
                            break;
                        }
                        *current += 1;
                    }
                    StepResult::Split(pc) => {
                        let current = splits.entry(pc).or_insert(1);
                        if *current <= MAX_REVISITS {
                            self.stack.push(vm.copy(pc));
                        }
                        *current += 1;
                    }
                    _ => {}
                }
            }
        }
    }
}

fn convert_to_address(v: &AbstractValueRef) -> Option<Address> {
    match &**v {
        AbstractValue::AddressConst(v) => Some(v.clone()),
        AbstractValue::CallResult {
            op: _,
            gas: _,
            address,
            value: _,
            selector: _,
            args: _,
        } => {
            return convert_to_address(&address);
        }
        _ => None,
    }
}

pub fn perform_analysis(
    bytecode: &Bytecode,
    context: &AnalysisContext,
) -> eyre::Result<(Vec<AnalyzedStoragesSlot>, Vec<Address>)> {
    let start = std::time::Instant::now();

    let bytes = &bytecode.bytes()[..];
    let contract_instructions = disassemble_bytes(bytes);

    let add_slot = move |v: &mut Vec<AnalyzedStoragesSlot>, slot: &[u8]| {
        let slot = U256::from_be_slice(slot);
        v.push(AnalyzedStoragesSlot::Slot(slot, SlotType::Unknown));
    };

    if contract_instructions.len() <= 3 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut slots = Vec::<AnalyzedStoragesSlot>::with_capacity(64);

    let mut jump_dests = HashMap::<usize, usize>::new();

    for i in 0..contract_instructions.len() {
        let ins0 = &contract_instructions[i];
        if i < contract_instructions.len() - 3 {
            let ins1 = &contract_instructions[i + 1];
            let ins2 = &contract_instructions[i + 2];
            if ins1.opcode == Opcode::PUSH0 {
                match ins2.opcode {
                    Opcode::SSTORE | Opcode::SLOAD => {
                        slots.push(AnalyzedStoragesSlot::Slot(U256::ZERO, SlotType::Unknown));
                    }
                    _ => {}
                }
            } else if ins1.opcode.is_push() {
                match ins2.opcode {
                    Opcode::SSTORE | Opcode::SLOAD => {
                        add_slot(&mut slots, ins1.input);
                    }
                    _ => {}
                }
            } else if ins0.opcode.is_push() && ins1.opcode == Opcode::DUP1 {
                match ins2.opcode {
                    Opcode::SSTORE | Opcode::SLOAD => {
                        add_slot(&mut slots, ins0.input);
                    }
                    _ => {}
                }
            }
        }
        if ins0.opcode == Opcode::JUMPDEST {
            jump_dests.insert(ins0.offset as usize, i);
        }
    }

    let (external_refs, storage_slots, type_masks) = {
        let program = contract_instructions;
        let program_bytes = bytes;
        let jump_dests = Rc::new(jump_dests);
        let mut vm = AbstractVM::new(program_bytes, &program, jump_dests.clone(), 0);
        vm.run(&context);

        (
            vm.analysis.external_contracts,
            vm.analysis.storage_slots,
            vm.analysis.slot_mask_size,
        )
    };

    let external_refs = external_refs
        .into_iter()
        .map(|v| reduce_externals(&v))
        .collect::<Vec<_>>();

    let contract_refs = external_refs
        .iter()
        .map(|v| convert_to_address(v))
        .flatten()
        .unique()
        .collect::<Vec<_>>();

    let cont_refs = external_refs
        .into_iter()
        .map(|v| convert(&v, &vec![], 0))
        .flatten()
        .unique()
        .collect::<Vec<AnalyzedStoragesSlot>>();
    let mut result: HashMap<U256, AnalyzedStoragesSlot> = HashMap::new();
    for slot in slots
        .iter()
        .chain(storage_slots.iter())
        .chain(cont_refs.iter())
    {
        let mut slot = slot.clone();
        let slot_index = slot.get_slot();
        let mask_type = type_masks.get(&slot_index);
        if let Some(mask_type) = mask_type {
            slot = slot.promote_type(mask_type);
        }

        match result.entry(slot_index) {
            Entry::Occupied(mut entry) => {
                entry.insert(entry.get().promote(&slot));
            }
            Entry::Vacant(entry) => {
                entry.insert(slot.clone());
            }
        }
    }
    let out_slots = result
        .into_values()
        .sorted_by_key(|v| v.get_slot())
        .collect::<Vec<_>>();

    log::debug!(target: LOGGER_TARGET_MAIN, "analysis took {:?}", start.elapsed());
    return Ok((out_slots, contract_refs));
}

#[cfg(test)]
mod tests {
    use alloy::{eips::BlockId, hex};
    use alloy_provider::Provider;
    use revm::primitives::{Bytecode, TxKind};

    use crate::utils::provider_from_string;

    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn it_works() {
        pretty_env_logger::init();
        // let test_addr = Address::from(hex!("bf1c0206de440b2cf76ea4405e1dbf2fc227a463"));
        // let test_addr = Address::from(hex!("784955641292b0014bc9ef82321300f0b6c7e36d"));
        // let test_addr = Address::from(hex!("ac3E018457B222d93114458476f3E3416Abbe38F"));

        // let test_addr = Address::from(hex!("7effd7b47bfd17e52fb7559d3f924201b9dbff3d"));
        // let test_addr = Address::from(hex!("43506849d7c04f9138d1a2050bbf3a0c054402dd"));
        // let test_addr = Address::from(hex!("BBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb"));
        // let test_addr = Address::from(hex!("2F50D538606Fa9EDD2B11E2446BEb18C9D5846bB"));
        let test_addr = Address::from(hex!("DbC0cE2321B76D3956412B36e9c0FA9B0fD176E7"));
        // let test_addr = Address::from(hex!("0aDc69041a2B086f8772aCcE2A754f410F211bed"));

        // let test_addr = Address::from(hex!("4da27a545c0c5B758a6BA100e3a049001de870f5"));
        // let test_addr: Address = Address::from(hex!("5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"));
        let url = std::env::var_os("PROVIDER").unwrap();
        let provider = provider_from_string(&url.to_string_lossy().to_string())
            .await
            .unwrap();

        let code = Bytecode::new_raw(provider.get_code_at(test_addr).await.unwrap());

        let block = provider
            .get_block(
                BlockId::latest(),
                alloy::rpc::types::BlockTransactionsKind::Hashes,
            )
            .await
            .unwrap()
            .unwrap();

        let context = AnalysisContext::from(&block)
            .with_contract_address(test_addr)
            .with_chain_id(U256::from(1));

        match perform_analysis(&code, &context) {
            Ok(data) => {
                println!("Analyzed");
                println!("Slots: {}", data.0.iter().join("\n"));
                println!("Addresses: {}", data.1.iter().join("\n"));
            }
            Err(e) => {
                println!("Failed to analyze {}", e);
            }
        };
    }
}
