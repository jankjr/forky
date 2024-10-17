use alloy::{hex, primitives::U160};
use evm_disassembler::{Opcode, Operation};
use itertools::Itertools;
use match_deref::match_deref;
use revm::{
    interpreter::{
        instructions::i256::{i256_cmp, i256_div, i256_mod},
        InstructionResult,
    },
    primitives::{Address, Bytecode, U256},
};

use std::{
    borrow::BorrowMut,
    collections::{hash_map::Entry, HashMap, HashSet},
    fmt::{Debug, Display},
    hash::Hash,
    rc::Rc,
};

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

type HeapRef<T> = Rc<T>;

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

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum SlotType {
    Address,
    Uint,
}

impl Display for SlotType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl SlotType {
    fn promote(&self, other: &SlotType) -> SlotType {
        match (self, other) {
            (SlotType::Address, _) => SlotType::Address,
            (_, SlotType::Address) => SlotType::Address,
            _ => SlotType::Uint,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum AbstractValue {
    Calldata(usize),
    Const(U256),
    AddressConst(Address),
    OpcodeResult(Opcode),
    CREATE(
        HeapRef<AbstractValue>,
        HeapRef<AbstractValue>,
        HeapRef<AbstractValue>,
    ),
    CREATE2(
        HeapRef<AbstractValue>,
        HeapRef<AbstractValue>,
        HeapRef<AbstractValue>,
        HeapRef<AbstractValue>,
    ),
    CallResult {
        op: Opcode,
        gas: HeapRef<AbstractValue>,
        address: HeapRef<AbstractValue>,
        value: Option<HeapRef<AbstractValue>>,
        selector: Option<HeapRef<AbstractValue>>,
        args: Vec<HeapRef<AbstractValue>>,
    },
    UnaryOpResult(Opcode, HeapRef<AbstractValue>),
    BinOpResult(Opcode, HeapRef<AbstractValue>, HeapRef<AbstractValue>),
    TertiaryOpResult(
        Opcode,
        HeapRef<AbstractValue>,
        HeapRef<AbstractValue>,
        HeapRef<AbstractValue>,
    ),
    StorageArray(HeapRef<AbstractValue>),
    StorageSlot(SlotType, U256),
    Mapping(SlotType, HeapRef<AbstractValue>),
    AddressRef(HeapRef<AbstractValue>),
}

type AbstractValueRef = HeapRef<AbstractValue>;

static ADDR_MASK: U256 = U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffffffffff"));

// static SIZE_MASKS: [U256; 31] = [
//     U256::from_be_slice(&hex!("ff")),
//     U256::from_be_slice(&hex!("ffff")),
//     U256::from_be_slice(&hex!("ffffff")),
//     U256::from_be_slice(&hex!("ffffffff")),
//     U256::from_be_slice(&hex!("ffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffffffffffffffffffff")),
//     U256::from_be_slice(&hex!(
//         "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
//     )),
//     U256::from_be_slice(&hex!(
//         "ffffffffffffffffffffffffffffffffffffffffffffffffffffff"
//     )),
//     U256::from_be_slice(&hex!(
//         "ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
//     )),
//     U256::from_be_slice(&hex!(
//         "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
//     )),
//     U256::from_be_slice(&hex!(
//         "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
//     )),
//     U256::from_be_slice(&hex!(
//         "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
//     )),
// ];

impl ToString for AbstractValue {
    fn to_string(&self) -> String {
        match self {
            AbstractValue::AddressRef(inner) => format!("Address({})", inner.to_string()),
            AbstractValue::AddressConst(inner) => format!("{}", inner.to_string()),
            AbstractValue::Mapping(arg_t, slot) => {
                format!("mapping({:?} => {})", arg_t, slot.to_string())
            }
            AbstractValue::CallResult {
                op,
                gas: _,
                address,
                value,
                selector,
                args,
            } => {
                let sel = match selector {
                    Some(v) => match &**v {
                        AbstractValue::Const(v) => {
                            let selector_bytes = (v.as_limbs()[0] >> 4) as u32;
                            format!(":{}", hex::encode(selector_bytes.to_le_bytes()))
                        }
                        _ => format!(":{:?}", v),
                    },
                    None => ":?".to_string(),
                };

                match value {
                    Some(value) => format!(
                        "{:?}{}.{}[value:{:?}]({})",
                        op,
                        sel,
                        address.to_string(),
                        value,
                        args.iter().map(|v| v.to_string()).join(", ")
                    ),
                    None => format!(
                        "{:?}{}.{}({})",
                        op,
                        sel,
                        address.to_string(),
                        args.iter().map(|v| v.to_string()).join(", ")
                    ),
                }
            }
            _ => format!("{:?}", self),
        }
    }
}
impl AbstractValue {
    fn bin(op: Opcode, a: AbstractValueRef, b: AbstractValueRef) -> AbstractValueRef {
        let v = match_deref! {
            match (op, &a, &b) {
                (Opcode::ADD, Deref@AbstractValue::Const(a), Deref@AbstractValue::Const(b)) => {
                    AbstractValue::Const(a.wrapping_add(*b))
                }
                (Opcode::ADD, Deref@AbstractValue::Const(a), Deref@AbstractValue::BinOpResult(Opcode::ADD, Deref@AbstractValue::Const(b), exp)) => {
                    AbstractValue::BinOpResult(Opcode::ADD, AbstractValue::val(&a.wrapping_add(*b)), exp.clone())
                }
                (Opcode::AND, Deref@AbstractValue::Const(a), Deref@AbstractValue::Const(b)) => {
                    return AbstractValue::val(&(a & b));
                }
                (Opcode::AND, exp, Deref@AbstractValue::Const(v)) => {
                    return AbstractValue::bin(Opcode::AND, AbstractValue::val(v), exp.clone());
                }
                (Opcode::DIV, exp, Deref@AbstractValue::Const(v)) => {
                    if v.eq(&U256::from(1)) {
                        return exp.clone();
                    }
                    AbstractValue::BinOpResult(op, a.clone(), b.clone())
                }
                (Opcode::AND, Deref@AbstractValue::Const(v), Deref@AbstractValue::AddressRef(inner)) => {
                    if v.eq(&ADDR_MASK) {
                        return HeapRef::new(AbstractValue::AddressRef(inner.clone()))
                    }
                    AbstractValue::BinOpResult(Opcode::AND, AbstractValue::val(v), HeapRef::new(AbstractValue::AddressRef(inner.clone())))
                }
                (Opcode::AND, Deref@AbstractValue::Const(v), exp) => {
                    if v.eq(&ADDR_MASK) {
                        return HeapRef::new(AbstractValue::AddressRef(exp.clone()))
                    }
                    AbstractValue::BinOpResult(Opcode::AND, AbstractValue::val(v), exp.clone())
                }
                (Opcode::ADD, exp, Deref@AbstractValue::Const(b)) => {
                    return AbstractValue::bin(Opcode::ADD, AbstractValue::val(b), exp.clone());
                }

                (Opcode::MUL, Deref@AbstractValue::Const(_), Deref@AbstractValue::Const(_)) => {
                    AbstractValue::BinOpResult(op, a, b)
                }
                (Opcode::MUL, exp, Deref@AbstractValue::Const(b)) => {
                    AbstractValue::BinOpResult(op, AbstractValue::val(b), exp.clone())
                }

                _ => AbstractValue::BinOpResult(op, a.clone(), b.clone())
            }
        };

        HeapRef::new(v)
    }
    fn map(input: SlotType, a: AbstractValueRef) -> AbstractValueRef {
        HeapRef::new(AbstractValue::Mapping(input, a.clone()))
    }
    fn array(a: AbstractValueRef) -> AbstractValueRef {
        HeapRef::new(AbstractValue::StorageArray(a.clone()))
    }
    fn unary(op: Opcode, a: AbstractValueRef) -> AbstractValueRef {
        HeapRef::new(AbstractValue::UnaryOpResult(op, a.clone()))
    }
    fn op(op: Opcode) -> AbstractValueRef {
        HeapRef::new(AbstractValue::OpcodeResult(op))
    }
    fn val(a: &U256) -> AbstractValueRef {
        HeapRef::new(AbstractValue::Const(*a))
    }

    fn ext_call(
        op: Opcode,
        gas: AbstractValueRef,
        address: AbstractValueRef,
        value: AbstractValueRef,
        selector: Option<AbstractValueRef>,
        args: Vec<AbstractValueRef>,
    ) -> AbstractValue {
        AbstractValue::CallResult {
            op,
            gas,
            address: reduce_externals(&address),
            value: Some(value),
            selector,
            args,
        }
    }

    fn static_or_delegate_call(
        op: Opcode,
        gas: AbstractValueRef,
        address: AbstractValueRef,
        selector: Option<AbstractValueRef>,
        args: Vec<AbstractValueRef>,
    ) -> AbstractValue {
        AbstractValue::CallResult {
            op,
            gas,
            address: reduce_externals(&address),
            value: None,
            selector,
            args,
        }
    }
}

pub fn decode_operation(
    bytes: &mut dyn ExactSizeIterator<Item = u8>,
    cur_offset: u32,
) -> eyre::Result<(Operation, u32)> {
    let encoded_opcode = bytes.next().expect("Unexpected end of input");
    let num_bytes = match encoded_opcode {
        0x60..=0x7f => encoded_opcode - 0x5f,
        _ => 0,
    };

    let mut new_offset = cur_offset + 1;
    let opcode = Opcode::from_byte(encoded_opcode);
    let mut operation = Operation::new(opcode, cur_offset);
    if num_bytes > 0 {
        new_offset += num_bytes as u32;
        operation = operation.with_bytes(num_bytes, bytes)?
    };
    Ok((operation, new_offset))
}

pub fn disassemble_bytes(bytes: Vec<u8>) -> Vec<Operation> {
    let mut operations = Vec::new();
    let mut new_operation: Operation;
    let mut offset = 0;
    let mut bytes_iter = bytes.into_iter();
    while bytes_iter.len() > 0 {
        (new_operation, offset) = match decode_operation(&mut bytes_iter, offset) {
            Ok((operation, new_offset)) => (operation, new_offset),
            Err(e) => {
                break;
            }
        };
        operations.push(new_operation);
    }
    operations
}

fn is_push(opcode: Opcode) -> bool {
    match opcode {
        Opcode::PUSH0
        | Opcode::PUSH1
        | Opcode::PUSH2
        | Opcode::PUSH3
        | Opcode::PUSH4
        | Opcode::PUSH5
        | Opcode::PUSH6
        | Opcode::PUSH7
        | Opcode::PUSH8
        | Opcode::PUSH9
        | Opcode::PUSH10
        | Opcode::PUSH11
        | Opcode::PUSH12
        | Opcode::PUSH13
        | Opcode::PUSH14
        | Opcode::PUSH15
        | Opcode::PUSH16
        | Opcode::PUSH17
        | Opcode::PUSH18
        | Opcode::PUSH19
        | Opcode::PUSH20
        | Opcode::PUSH21
        | Opcode::PUSH22
        | Opcode::PUSH23
        | Opcode::PUSH24
        | Opcode::PUSH25
        | Opcode::PUSH26
        | Opcode::PUSH27
        | Opcode::PUSH28
        | Opcode::PUSH29
        | Opcode::PUSH30
        | Opcode::PUSH31
        | Opcode::PUSH32 => true,
        _ => false,
    }
}

macro_rules! debug_unreachable {
    ($($t:tt)*) => {
        if cfg!(debug_assertions) {
            unreachable!($($t)*);
        } else {
            unsafe { core::hint::unreachable_unchecked() };
        }
    };
}

macro_rules! assume {
    ($e:expr $(,)?) => {
        if !$e {
            debug_unreachable!(stringify!($e));
        }
    };

    ($e:expr, $($t:tt)+) => {
        if !$e {
            debug_unreachable!($($t)+);
        }
    };
}

pub const STACK_LIMIT: usize = 1024;
#[derive(Debug)]
struct AbstractStack {
    data: Vec<HeapRef<AbstractValue>>,
}
impl AbstractStack {
    pub fn copy(&self) -> Self {
        let mut d = Vec::with_capacity(STACK_LIMIT);
        for v in self.data.iter() {
            d.push(v.clone());
        }
        Self { data: d }
    }
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(STACK_LIMIT),
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Result<HeapRef<AbstractValue>, InstructionResult> {
        match self.data.pop() {
            Some(v) => Ok(v.clone()),
            None => Err(InstructionResult::StackUnderflow),
        }
    }

    /// Push a new value onto the stack.
    ///
    /// If it will exceed the stack limit, returns `StackOverflow` error and leaves the stack
    /// unchanged.
    #[inline]
    pub fn push(&mut self, value: HeapRef<AbstractValue>) -> Result<(), InstructionResult> {
        // Allows the compiler to optimize out the `Vec::push` capacity check.
        assume!(self.data.capacity() == STACK_LIMIT);
        if self.data.len() == STACK_LIMIT {
            return Err(InstructionResult::StackOverflow);
        }
        self.data.push(value);
        Ok(())
    }

    #[inline]
    pub fn push_uint(&mut self, value: u64) -> Result<(), InstructionResult> {
        let out = AbstractValue::Const(U256::from(value));
        self.push(HeapRef::new(out))
    }

    #[inline]
    pub fn dup(&mut self, n: usize) -> Result<(), InstructionResult> {
        assume!(n > 0, "attempted to dup 0");
        if n > self.data.len() {
            // println!("dup: stack underflow");
            return Err(InstructionResult::StackUnderflow);
        }
        let element = self.data.get(self.data.len() - n);
        match element {
            None => Err(InstructionResult::StackOverflow),
            Some(v) => {
                self.data.push(v.clone());
                Ok(())
            }
        }
    }

    /// Swaps the topmost value with the `N`th value from the top.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    #[inline(always)]
    pub fn swap(&mut self, nth: usize) -> Result<(), InstructionResult> {
        let top = self.data.len();
        if nth >= top {
            return Err(InstructionResult::StackOverflow);
        }
        self.data.swap(top - 1, top - nth - 1);
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum AbstractMemoryValue {
    Abstract {
        size: usize,
        value: HeapRef<AbstractValue>,
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
        offset: &HeapRef<AbstractValue>,
        size: &HeapRef<AbstractValue>,
    ) -> (Option<HeapRef<AbstractValue>>, Vec<HeapRef<AbstractValue>>) {
        let (mut offset, mut size): (usize, usize) = match_deref::match_deref! {
            match (offset, size) {
                (Deref @ AbstractValue::Const(offset), Deref @ AbstractValue::Const(size)) => (offset.to(), size.to()),
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
            value: HeapRef::new(AbstractValue::BinOpResult(
                Opcode::MLOAD,
                HeapRef::new(AbstractValue::Const(U256::from(offset))),
                HeapRef::new(AbstractValue::Const(U256::from(read_size))),
            )),
        };
    }

    #[inline]
    fn store(&mut self, offset: usize, value: AbstractMemoryValue) {
        self.backing.insert(offset, value);
    }

    #[inline]
    fn mstore(&mut self, offset: HeapRef<AbstractValue>, value: HeapRef<AbstractValue>) {
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
    fn mstore8(&mut self, offset: HeapRef<AbstractValue>, value: HeapRef<AbstractValue>) {
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
    fn get_word(&self, offset: usize) -> Option<HeapRef<AbstractValue>> {
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
    fn load(&self, offset_v: HeapRef<AbstractValue>) -> Option<HeapRef<AbstractValue>> {
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

fn infer_type(v: &HeapRef<AbstractValue>) -> SlotType {
    match **v {
        AbstractValue::AddressRef(_) => SlotType::Address,
        _ => SlotType::Uint,
    }
}

fn reduce_externals(v: &HeapRef<AbstractValue>) -> HeapRef<AbstractValue> {
    use AbstractValue::*;

    HeapRef::new(match_deref! {
        match v {
            Deref @ Const(v) => AddressConst(Address::from(U160::from(*v))),
            v => return reduce(v)
        }
    })
}

fn reduce(v: &HeapRef<AbstractValue>) -> HeapRef<AbstractValue> {
    use AbstractValue::*;
    use Opcode::*;

    match_deref! {
        match v {
            Deref @ AddressRef(inner) => return HeapRef::new(AddressRef(reduce(&inner))),
            Deref @ UnaryOpResult(SLOAD, Deref @ Const(v)) => return HeapRef::new(StorageSlot(SlotType::Uint, *v)),
            Deref @ UnaryOpResult(SLOAD, Deref @ UnaryOpResult(SHA3, Deref@Const(v))) => return HeapRef::new(StorageArray(HeapRef::new(StorageSlot(SlotType::Uint, *v)))),
            Deref @ UnaryOpResult(SLOAD, Deref @ BinOpResult(SHA3, arg1, Deref @ BinOpResult(SHA3, arg0, Deref @ Const(v)))) => return AbstractValue::map(infer_type(arg0), AbstractValue::map(infer_type(arg1), HeapRef::new(StorageSlot(SlotType::Uint, *v)))),
            Deref @ UnaryOpResult(SLOAD, Deref @ BinOpResult(SHA3, arg0, Deref @ Const(v))) => return AbstractValue::map(infer_type(arg0), HeapRef::new(StorageSlot(SlotType::Uint, *v))),
            Deref @ UnaryOpResult(SLOAD, Deref @ BinOpResult(ADD, Deref@Const(_), Deref@UnaryOpResult(SHA3, exp))) => return AbstractValue::array(reduce(exp)),
            Deref @ UnaryOpResult(SLOAD, Deref @ BinOpResult(ADD, Deref@Const(_), Deref@BinOpResult(SHA3, arg0, Deref@Const(v)))) => return AbstractValue::map(infer_type(arg0), HeapRef::new(StorageSlot(SlotType::Uint, *v))),
            Deref @ UnaryOpResult(SLOAD, Deref @ BinOpResult(ADD, Deref@Const(_), Deref@BinOpResult(SHA3, arg1, Deref @ BinOpResult(SHA3, arg0, Deref @ Const(v))))) => return AbstractValue::map(infer_type(arg0), AbstractValue::map(infer_type(arg1), HeapRef::new(StorageSlot(SlotType::Uint, *v)))),
            Deref @ Const(v) => return HeapRef::new(StorageSlot(SlotType::Uint, *v)),
            _ => {}
        }
    };

    return v.clone();
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum AnalyzedStoragesSlot {
    Slot(U256, SlotType),
    Array(U256, SlotType),
    Mapping(U256, Vec<SlotType>, SlotType),
}

impl AnalyzedStoragesSlot {
    fn promote(&self, other: &AnalyzedStoragesSlot) -> Self {
        match (self, other) {
            (AnalyzedStoragesSlot::Slot(slot, typ), AnalyzedStoragesSlot::Array(_, other_typ)) => {
                AnalyzedStoragesSlot::Array(*slot, typ.promote(other_typ))
            }
            (
                AnalyzedStoragesSlot::Slot(slot, typ),
                AnalyzedStoragesSlot::Mapping(_, other_map, other_typ),
            ) => AnalyzedStoragesSlot::Mapping(*slot, other_map.clone(), typ.promote(other_typ)),
            (
                AnalyzedStoragesSlot::Array(slot, typ),
                AnalyzedStoragesSlot::Mapping(_, other_map, other_typ),
            ) => AnalyzedStoragesSlot::Mapping(*slot, other_map.clone(), typ.promote(other_typ)),
            _ => self.promote_type(&other.get_type()),
        }
    }
    fn promote_type(&self, new_type: &SlotType) -> AnalyzedStoragesSlot {
        match self {
            AnalyzedStoragesSlot::Slot(slot, typ) => {
                AnalyzedStoragesSlot::Slot(*slot, typ.promote(new_type))
            }
            AnalyzedStoragesSlot::Array(slot, typ) => {
                AnalyzedStoragesSlot::Array(*slot, typ.promote(new_type))
            }
            AnalyzedStoragesSlot::Mapping(slot, map, typ) => {
                AnalyzedStoragesSlot::Mapping(*slot, map.clone(), typ.promote(new_type))
            }
        }
    }

    pub fn get_slot(&self) -> U256 {
        match self {
            AnalyzedStoragesSlot::Slot(slot, _) => *slot,
            AnalyzedStoragesSlot::Array(slot, _) => *slot,
            AnalyzedStoragesSlot::Mapping(slot, __, _) => *slot,
        }
    }

    fn get_mapping(&self) -> Vec<SlotType> {
        match self {
            AnalyzedStoragesSlot::Mapping(_, map, _) => map.clone(),
            _ => vec![],
        }
    }

    pub fn get_type(&self) -> SlotType {
        match self {
            AnalyzedStoragesSlot::Slot(_, typ) => typ.clone(),
            AnalyzedStoragesSlot::Array(_, typ) => typ.clone(),
            AnalyzedStoragesSlot::Mapping(_, _, typ) => typ.clone(),
        }
    }
}

impl Display for AnalyzedStoragesSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalyzedStoragesSlot::Slot(slot, typ) => write!(f, "Slot({slot}, {typ:?})"),
            AnalyzedStoragesSlot::Array(slot, typ) => write!(f, "Slot({slot}, {typ:?}[])"),
            AnalyzedStoragesSlot::Mapping(slot, map, typ) => write!(
                f,
                "Slot({slot}, mapping({} => {}))",
                map.iter().join(" => "),
                typ
            ),
        }
    }
}

fn convert(v: &HeapRef<AbstractValue>) -> Option<AnalyzedStoragesSlot> {
    use AbstractValue::*;
    let out = match_deref! {
        match v {
            Deref @ StorageSlot(typ, slot) => AnalyzedStoragesSlot::Slot(*slot, typ.clone()),
            Deref @ AddressRef(inner) => return convert(inner).map(|v| v.promote_type(&SlotType::Address)),
            Deref @ StorageArray(slot) => {
                let slot = convert(slot)?;
                AnalyzedStoragesSlot::Array(slot.get_slot(), slot.get_type())
            }
            Deref @ Mapping(arg_type, slot) => {
                let slot = convert(slot)?;
                let mut mapping = vec![arg_type.clone()];
                mapping.extend(slot.get_mapping());

                AnalyzedStoragesSlot::Mapping(slot.get_slot(), mapping , slot.get_type())
            }
            _ => return None
        }
    };
    Some(out)
}

macro_rules! pop_or_break {
    ($stack:expr) => {{
        match $stack.pop() {
            Err(_) => {
                // println!("pop_or_break failed {:?}", e);
                return StepResult::Stop;
            }
            Ok(v) => v.clone(),
        }
    }};
}

macro_rules! handle_unary {
    ($stack:expr, $op:ident, $f:expr) => {
        let val = pop_or_break!($stack);
        match_deref::match_deref! {
            match &val {
                Deref @ AbstractValue::Const(v) => {
                    let out: U256 = U256::from($f(v));
                    push_or_break!($stack, HeapRef::new(AbstractValue::Const(out)));
                }
                _ => {
                    push_or_break_val!(
                        $stack,
                        AbstractValue::UnaryOpResult(Opcode::$op, val.clone())
                    );
                }
            }
        }
    };
}

macro_rules! push_or_break {
    ($stack:expr, $n:expr) => {{
        match $stack.push($n.clone()) {
            Err(_) => {
                return StepResult::Stop;
            }
            Ok(_) => {}
        }
    }};
}
macro_rules! push_or_break_val {
    ($stack:expr, $n:expr) => {{
        match $stack.push(HeapRef::new($n)) {
            Err(_) => {
                return StepResult::Stop;
            }
            Ok(_) => {}
        }
    }};
}

macro_rules! handle_bin {
    ($stack:expr, $op:expr, $opName:ident) => {
        let a = pop_or_break!($stack);
        let b = pop_or_break!($stack);
        match_deref::match_deref! {
            match (&a, &b) {
                (Deref@AbstractValue::Const(a), Deref@AbstractValue::Const(b)) => {
                    let res = U256::from($op(a.clone(), b.clone()));
                    // println!("{} {:?} {:?} -> {:?}", stringify!($opName), a, b, res);
                    push_or_break!($stack, HeapRef::new(AbstractValue::Const(res)));
                }
                _ => {
                    push_or_break!(
                        $stack,
                        AbstractValue::bin(Opcode::$opName, a.clone(), b.clone())
                    );
                }
            }
        }
    };
}

macro_rules! handle_shift_op {
    ($stack:expr, $op:expr, $opName:ident) => {
        let shift = pop_or_break!($stack);
        let value = pop_or_break!($stack);
        match_deref::match_deref! {
            match (&shift, &value) {
                (Deref @ AbstractValue::Const(shift), Deref @ AbstractValue::Const(value)) => {
                    let shift: usize = shift.to();
                    let res = U256::from($op(*value, shift));
                    // println!("{} {:?} {:?} -> {:?}", stringify!($opName), shift, value, res);
                    push_or_break!($stack, HeapRef::new(AbstractValue::Const(res)));
                }
                _ => {
                    push_or_break!(
                        $stack,
                        AbstractValue::bin(Opcode::$opName, shift.clone(), value.clone())
                    );
                }
            }
        }
    };
}

macro_rules! handle_bin_op {
    ($stack:expr, $op:tt, $opName:ident) => {
        let a = pop_or_break!($stack);
        let b = pop_or_break!($stack);
        match_deref::match_deref! {
            match (&a, &b) {
                (Deref@AbstractValue::Const(a), Deref@AbstractValue::Const(b)) => {
                    let res = U256::from(a.clone() $op b.clone());
                    // println!("{} {:?} {:?} -> {:?}", stringify!($opName), a, b, res);
                    push_or_break!($stack, HeapRef::new(AbstractValue::Const(res)));
                }
                _ => {
                    push_or_break!(
                        $stack,
                        AbstractValue::bin(Opcode::$opName, a.clone(), b.clone())
                    );
                }
            }
        }
    };
}

macro_rules! handle_bin_c {
    ($stack:expr, $op:expr, $opName:ident) => {
        let a = pop_or_break!($stack);
        let b = pop_or_break!($stack);

        match_deref::match_deref! {
            match (&a, &b) {
                (Deref @ AbstractValue::Const(a), Deref @ AbstractValue::Const(b)) => {
                    let res = if $op(&a.clone(), &b.clone()) {
                        U256::from(1u64)
                    } else {
                        U256::ZERO
                    };

                    // println!("{} {:?} {:?} -> {:?}", stringify!($opName), a, b, res);
                    push_or_break!($stack, HeapRef::new(AbstractValue::Const(res)));
                }
                (a, b) => {
                    push_or_break!($stack, HeapRef::new(AbstractValue::BinOpResult(Opcode::$opName, a.clone(), b.clone())));
                }
            }
        }
    };
}

macro_rules! handle_dup {
    ($stack:expr, $n:expr) => {
        match $stack.dup($n) {
            Err(_) => {
                return StepResult::Stop;
            }
            Ok(_) => {}
        }
    };
}

macro_rules! handle_swap {
    ($stack:expr, $n:expr) => {
        match $stack.swap($n) {
            Err(_) => {
                return StepResult::Stop;
            }
            Ok(_) => {}
        }
    };
}

enum StepResult {
    Ok,
    Stop,
    Split(usize),
}
struct AbstractVMInstance {
    program_bytes: Rc<Vec<u8>>,
    program: Rc<Vec<Operation>>,
    pc: usize,
    jump_dests: Rc<HashMap<usize, usize>>,
    stack: AbstractStack,
    memory: AbstractMemory,
    storage: HashMap<U256, HeapRef<AbstractValue>>,
    tmemory: HashMap<U256, HeapRef<AbstractValue>>,
    steps: usize,

    return_data_size: usize,

    halted: bool,
}
impl AbstractVMInstance {
    fn copy(&self, pc: usize) -> Self {
        Self {
            program_bytes: self.program_bytes.clone(),
            program: self.program.clone(),
            jump_dests: self.jump_dests.clone(),
            pc,
            stack: self.stack.copy(),
            memory: self.memory.copy(),
            storage: self.storage.clone(),
            tmemory: self.tmemory.clone(),
            halted: self.halted,
            return_data_size: self.return_data_size,
            steps: self.steps,
        }
    }
    fn new(
        program_bytes: Rc<Vec<u8>>,
        program: Rc<Vec<Operation>>,
        jump_dests: Rc<HashMap<usize, usize>>,
        pc: usize,
    ) -> Self {
        Self {
            program_bytes: program_bytes,
            program,
            pc,
            jump_dests,
            stack: AbstractStack::new(),
            memory: AbstractMemory::new(),
            storage: HashMap::new(),
            tmemory: HashMap::new(),
            steps: 0,
            return_data_size: 0,
            halted: false,
        }
    }

    pub fn step(&mut self, analysis: &mut Analysis) -> StepResult {
        if self.halted {
            return StepResult::Stop;
        }

        if self.steps > 300 {
            self.halted = true;
            return StepResult::Stop;
        }

        self.steps += 1;
        let prog = self.program.clone();
        let pc = self.pc;
        let ins = prog.get(pc);
        let ins = match ins {
            None => {
                return StepResult::Stop;
            }
            Some(v) => v,
        };
        // println!("{:?}", ins);
        let res = self.step_(ins, analysis);
        match &res {
            StepResult::Stop => {
                self.halted = true;
            }
            _ => {}
        }
        res
    }

    fn step_(&mut self, ins: &Operation, analysis: &mut Analysis) -> StepResult {
        let stack = &mut self.stack;
        let memory = &mut self.memory;
        let storage = &mut self.storage;
        let tmemory = &mut self.tmemory;
        let jump_dests = &self.jump_dests;

        match ins.opcode {
            Opcode::SELFDESTRUCT => {
                pop_or_break!(stack);
                return StepResult::Stop;
            }
            Opcode::STOP | Opcode::INVALID => {
                return StepResult::Stop;
            }
            Opcode::REVERT | Opcode::RETURN => {
                pop_or_break!(stack);
                pop_or_break!(stack);
                return StepResult::Stop;
            }
            Opcode::JUMP => {
                let byte_offset = pop_or_break!(stack);
                match_deref::match_deref! {
                    match &byte_offset {
                        Deref @ AbstractValue::Const(v) => {
                            let offset: usize = v.to();
                            if let Some(new_pc) = jump_dests.get(&offset) {
                                self.pc = *new_pc;
                                return StepResult::Ok;
                            } else {
                                return StepResult::Stop;
                            }
                        }
                        _ => {
                            return StepResult::Stop;
                        }
                    }
                };
            }
            Opcode::JUMPI => {
                let byte_offset = pop_or_break!(stack);
                let cond = pop_or_break!(stack);
                match_deref::match_deref! {
                    match (&cond, &byte_offset) {
                        (_, Deref @ AbstractValue::Const(byte_offset)) => {
                            let offset: usize = byte_offset.to();
                            if let Some(new_pc) = jump_dests.get(&offset) {
                                let split_pc = self.pc + 1;
                                self.pc = *new_pc;
                                return StepResult::Split(split_pc);
                            } else {
                                self.pc += 1;
                            }
                        }
                        (a, b) => {
                            self.pc += 1;
                        }
                    }
                }
                return StepResult::Ok;
            }
            _ => {}
        };
        match ins.opcode {
            Opcode::STATICCALL | Opcode::DELEGATECALL => {
                let gas = pop_or_break!(stack);
                let address = pop_or_break!(stack);
                analysis.external_contracts.push(address.clone());

                let args_offset = pop_or_break!(stack);
                let args_size = pop_or_break!(stack);
                let ret_offset = pop_or_break!(stack);
                let ret_size = pop_or_break!(stack);
                if let AbstractValue::Const(v) = *ret_size {
                    let size: usize = v.to();
                    self.return_data_size = size;
                }
                let (selector, args) = memory.load_args(&args_offset, &args_size);

                let res = HeapRef::new(AbstractValue::static_or_delegate_call(
                    ins.opcode, gas, address, selector, args,
                ));

                memory.mstore(ret_offset, res.clone());

                push_or_break!(stack, res.clone())
            }
            Opcode::CALL | Opcode::CALLCODE => {
                let gas = pop_or_break!(stack);
                let address = pop_or_break!(stack);

                analysis.external_contracts.push(address.clone());

                let value = pop_or_break!(stack);
                let args_offset = pop_or_break!(stack);
                let args_size = pop_or_break!(stack);
                let ret_offset = pop_or_break!(stack);
                let ret_size = pop_or_break!(stack);
                if let AbstractValue::Const(v) = *ret_size {
                    let size: usize = v.to();
                    self.return_data_size = size;
                }

                let (selector, args) = memory.load_args(&args_offset, &args_size);

                let res = HeapRef::new(AbstractValue::ext_call(
                    ins.opcode, gas, address, value, selector, args,
                ));
                memory.mstore(ret_offset, res.clone());
                push_or_break!(stack, res);
            }

            Opcode::CREATE => {
                let val = AbstractValue::CREATE(
                    pop_or_break!(stack),
                    pop_or_break!(stack),
                    pop_or_break!(stack),
                );
                push_or_break_val!(stack, val)
            }

            Opcode::CREATE2 => {
                let val = AbstractValue::CREATE2(
                    pop_or_break!(stack),
                    pop_or_break!(stack),
                    pop_or_break!(stack),
                    pop_or_break!(stack),
                );
                push_or_break_val!(stack, val)
            }

            Opcode::ADD => {
                handle_bin!(stack, U256::wrapping_add, ADD);
            }
            Opcode::SUB => {
                handle_bin!(stack, U256::wrapping_sub, SUB);
            }
            Opcode::MUL => {
                handle_bin!(stack, U256::wrapping_mul, MUL);
            }
            Opcode::DIV => {
                let a = pop_or_break!(stack);
                let b = pop_or_break!(stack);
                match_deref::match_deref! {
                    match (&a, &b) {
                        (Deref @ AbstractValue::Const(a), Deref @ AbstractValue::Const(b)) => {
                            if b.is_zero() {
                                push_or_break!(stack, AbstractValue::val(&a));
                            } else {
                                push_or_break!(stack, AbstractValue::val(&a.wrapping_div(*b)));
                            }
                        },
                        _ => {
                            push_or_break!(
                                stack,
                                AbstractValue::bin(Opcode::DIV, a.clone(), b.clone())
                            )
                        }
                    }
                };
            }
            Opcode::AND => {
                handle_bin_op!(stack, &, AND);

                let top = if let Some(top) = stack.data.last() {
                    top
                } else {
                    self.pc += 1;
                    return StepResult::Ok;
                };
                match_deref! {
                    match top {
                        Deref@AbstractValue::AddressRef(Deref@AbstractValue::UnaryOpResult(Opcode::SLOAD, _)) => {
                            analysis.external_contracts.push(top.clone());
                        }
                        Deref@AbstractValue::BinOpResult(Opcode::AND, Deref@AbstractValue::Const(_), Deref@AbstractValue::UnaryOpResult(Opcode::SLOAD, _)) => {
                            // for (i, size) in SIZE_MASKS.iter().enumerate() {
                            //     if mask.eq(size) {
                            //         break;
                            //     }
                            // }
                        }

                        _ => {}
                    }
                }
            }
            Opcode::SHL => {
                handle_shift_op!(stack, U256::wrapping_shl, SHL);
            }
            Opcode::SHR => {
                handle_shift_op!(stack, U256::wrapping_shr, SHR);
            }
            Opcode::MOD => {
                handle_bin!(stack, U256::wrapping_rem, MOD);
            }
            Opcode::SDIV => {
                handle_bin!(stack, i256_div, SDIV);
            }
            Opcode::EQ => {
                handle_bin_c!(stack, U256::eq, EQ);
            }
            Opcode::LT => {
                handle_bin_c!(stack, U256::lt, LT);
            }
            Opcode::GT => {
                handle_bin_c!(stack, U256::gt, GT);
            }
            Opcode::NOT => {
                let v = pop_or_break!(stack);
                let out = match_deref! {
                    match &v {
                        Deref @ AbstractValue::Const(v) => {
                            AbstractValue::Const(!v)
                        }
                        _ => AbstractValue::UnaryOpResult(Opcode::NOT, v)
                    }
                };
                push_or_break_val!(stack, out)
            }
            Opcode::ISZERO => {
                handle_unary!(stack, ISZERO, U256::is_zero);
            }
            Opcode::POP => {
                pop_or_break!(stack);
            }
            Opcode::OR => {
                handle_bin_op!(stack, |, OR);
            }
            Opcode::XOR => {
                handle_bin_op!(stack, ^, XOR);
            }
            Opcode::SMOD => {
                handle_bin!(stack, i256_mod, SMOD);
            }

            Opcode::MULMOD | Opcode::ADDMOD => {
                let a = pop_or_break!(stack);
                let b = pop_or_break!(stack);
                let c = pop_or_break!(stack);
                match_deref::match_deref! {
                    match (&a, &b, &c) {
                        (
                            Deref @ AbstractValue::Const(a),
                            Deref @ AbstractValue::Const(b),
                            Deref @ AbstractValue::Const(c),
                        ) => {
                            if ins.opcode == Opcode::MULMOD {
                                push_or_break!(
                                    stack,
                                    HeapRef::new(AbstractValue::Const(U256::mul_mod(a.clone(), b.clone(), c.clone())))
                                );
                            } else {
                                push_or_break!(
                                    stack,
                                    HeapRef::new(AbstractValue::Const(U256::add_mod(a.clone(), b.clone(), c.clone())))
                                );
                            }
                        }
                        _ => {
                            push_or_break!(
                                stack,
                                HeapRef::new(AbstractValue::TertiaryOpResult(ins.opcode, a, b, c))
                            );
                        }
                    }
                }
            }
            Opcode::EXP => {
                handle_bin!(stack, U256::pow, EXP);
            }
            Opcode::SIGNEXTEND => {
                let sign_extend = sign_extend;
                handle_bin!(stack, sign_extend, SIGNEXTEND);
            }
            Opcode::SLT | Opcode::SGT => {
                let order = if ins.opcode == Opcode::SLT {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                };
                let a = pop_or_break!(stack);
                let b = pop_or_break!(stack);
                match_deref::match_deref! {
                    match (&a, &b) {
                        (Deref @ AbstractValue::Const(a), Deref @ AbstractValue::Const(b)) => {
                            push_or_break!(
                                stack,
                                HeapRef::new(AbstractValue::Const(U256::from(
                                    i256_cmp(a, b) == order
                                )))
                            );
                        }
                        _ => {
                            push_or_break!(
                                stack,
                                HeapRef::new(AbstractValue::BinOpResult(ins.opcode, a, b))
                            )
                        }
                    }
                }
            }
            Opcode::BYTE => {
                let i = pop_or_break!(stack);
                let w = pop_or_break!(stack);
                match_deref::match_deref! {
                    match (&w, &i) {
                        (Deref @ AbstractValue::Const(w), Deref @ AbstractValue::Const(i)) => {
                            let i: usize = i.to();
                            let b = w.byte(31 - i);
                            push_or_break!(stack, HeapRef::new(AbstractValue::Const(U256::from(b))));
                        }
                        _ => {
                            push_or_break!(
                                stack,
                                HeapRef::new(AbstractValue::BinOpResult(Opcode::BYTE, w, i))
                            );
                        }
                    }
                }
            }
            Opcode::SAR => {
                handle_shift_op!(stack, U256::arithmetic_shr, SAR);
            }
            Opcode::SHA3 => {
                let offset = pop_or_break!(stack);
                let size = pop_or_break!(stack);
                let exact = match_deref::match_deref! {
                    match (&offset, &size) {
                        (Deref @ AbstractValue::Const(offset), Deref @ AbstractValue::Const(size)) => {
                            if offset.gt(&U256::from(10000)) || size.gt(&U256::from(96)) {
                                None
                            } else {
                                let offset: usize = offset.to();
                                let size: usize = size.to();
                                let out = if size == 32 {
                                    memory.get_word(offset).map(|v|HeapRef::new(AbstractValue::UnaryOpResult(Opcode::SHA3, v)))
                                } else if size == 64 {
                                    if let (Some(v0), Some(v1)) = (memory.get_word(offset), memory.get_word(offset+32)) {
                                        Some(HeapRef::new(AbstractValue::BinOpResult(Opcode::SHA3, v0, v1)))
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
                    push_or_break!(stack, v);
                } else {
                    let value = memory.load(offset.clone());
                    match value {
                        Some(v) => {
                            push_or_break!(
                                stack,
                                HeapRef::new(AbstractValue::UnaryOpResult(Opcode::SHA3, v))
                            );
                        }
                        None => {
                            push_or_break!(
                                stack,
                                HeapRef::new(AbstractValue::BinOpResult(
                                    Opcode::SHA3,
                                    offset,
                                    size
                                ))
                            );
                        }
                    }
                }
            }
            Opcode::CALLDATALOAD => {
                let offset = pop_or_break!(stack);
                match_deref::match_deref! {
                    match &offset {
                        Deref @ AbstractValue::Const(offset) => {
                            push_or_break_val!(stack, AbstractValue::Calldata(offset.to()));
                        }
                        _ => {
                            push_or_break_val!(
                                stack,
                                AbstractValue::UnaryOpResult(Opcode::CALLDATALOAD, offset)
                            )
                        }
                    }
                }
            }
            Opcode::CODESIZE => {
                push_or_break_val!(stack, AbstractValue::Const(U256::from(self.program.len())))
            }
            Opcode::BALANCE | Opcode::EXTCODESIZE | Opcode::EXTCODEHASH => {
                let address = pop_or_break!(stack);
                analysis.external_contracts.push(address.clone());
                push_or_break_val!(stack, AbstractValue::UnaryOpResult(ins.opcode, address))
            }
            Opcode::EXTCODECOPY => {
                let address = pop_or_break!(stack);
                analysis.external_contracts.push(address.clone());
                pop_or_break!(stack);
                let offset = pop_or_break!(stack);
                pop_or_break!(stack);
                memory.mstore(
                    offset,
                    HeapRef::new(AbstractValue::OpcodeResult(Opcode::EXTCODECOPY)),
                );
            }
            Opcode::RETURNDATACOPY | Opcode::CALLDATACOPY | Opcode::CODECOPY => {
                pop_or_break!(stack);
                let offset = pop_or_break!(stack);
                pop_or_break!(stack);
                memory.mstore(
                    offset,
                    HeapRef::new(AbstractValue::OpcodeResult(ins.opcode)),
                );
            }
            Opcode::BLOBHASH => {
                pop_or_break!(stack);
                push_or_break_val!(stack, AbstractValue::OpcodeResult(Opcode::BLOBHASH));
            }
            Opcode::BLOBBASEFEE => {
                push_or_break_val!(stack, AbstractValue::OpcodeResult(Opcode::BLOBBASEFEE));
            }
            Opcode::BLOCKHASH => {
                let block_number = pop_or_break!(stack);
                push_or_break_val!(
                    stack,
                    AbstractValue::UnaryOpResult(Opcode::BLOCKHASH, block_number)
                );
            }
            Opcode::RETURNDATASIZE => {
                stack.push_uint(self.return_data_size as u64).unwrap();
            }
            Opcode::CHAINID => {
                stack.push_uint(1).unwrap();
            }
            Opcode::GAS
            | Opcode::CALLVALUE
            | Opcode::CALLDATASIZE
            | Opcode::GASPRICE
            | Opcode::BASEFEE
            | Opcode::ADDRESS
            | Opcode::ORIGIN
            | Opcode::CALLER
            | Opcode::COINBASE
            | Opcode::DIFFICULTY
            | Opcode::GASLIMIT
            | Opcode::SELFBALANCE
            | Opcode::NUMBER => {
                push_or_break_val!(stack, AbstractValue::OpcodeResult(ins.opcode))
            }
            Opcode::TIMESTAMP => {
                let timestamp_in_seconds = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                push_or_break_val!(
                    stack,
                    AbstractValue::Const(U256::from(timestamp_in_seconds))
                );
            }
            Opcode::MLOAD => {
                let offset = pop_or_break!(stack);
                match memory.load(offset.clone()) {
                    Some(v) => {
                        // println!("MLOAD({:?}) = {:?}", offset, v);
                        push_or_break!(stack, v)
                    }
                    None => {
                        let res = AbstractValue::UnaryOpResult(Opcode::MLOAD, offset.clone());
                        // println!("Opcode::MLOAD({:?}) = {:?}", offset, res);
                        push_or_break_val!(stack, res)
                    }
                }
            }
            Opcode::MSTORE => {
                let (offset, value) = (pop_or_break!(stack), pop_or_break!(stack));
                // println!("MSTORE({:?}, {:?})", offset, value);
                memory.mstore(offset, value);
            }
            Opcode::MSTORE8 => {
                memory.mstore8(pop_or_break!(stack), pop_or_break!(stack));
            }
            Opcode::SLOAD => {
                let offset = pop_or_break!(stack);
                analysis
                    .storage_slots
                    .push(AbstractValue::unary(Opcode::SLOAD, offset.clone()));
                let storage_value = match_deref::match_deref! {
                    match &offset {
                        Deref @ AbstractValue::Const(index) => {
                            let out = match storage.get(&index) {
                                None => Rc::new(AbstractValue::UnaryOpResult(Opcode::SLOAD, offset)),
                                Some(v) => v.clone(),
                            };
                            out
                        }
                        _ => Rc::new(AbstractValue::UnaryOpResult(Opcode::SLOAD, offset))
                    }
                };

                push_or_break!(stack, storage_value);
            }
            Opcode::SSTORE => {
                let offset = pop_or_break!(stack);
                analysis
                    .storage_slots
                    .push(AbstractValue::unary(Opcode::SLOAD, offset.clone()));
                pop_or_break!(stack);
            }
            Opcode::PC => {
                push_or_break_val!(stack, AbstractValue::Const(U256::from(ins.offset)));
            }
            Opcode::MSIZE => {
                push_or_break_val!(stack, AbstractValue::Const(U256::from(memory.len())));
            }
            Opcode::MCOPY => {
                let dest = pop_or_break!(stack);
                let offset = pop_or_break!(stack);
                let size = pop_or_break!(stack);
                match_deref::match_deref! {
                    match (&dest, &offset, &size) {
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
                let offset_v = pop_or_break!(stack);

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
                push_or_break!(stack, res);
            }
            Opcode::TSTORE => {
                let offset = pop_or_break!(stack);
                let value = pop_or_break!(stack);
                match_deref::match_deref! {
                    match &offset {
                        Deref @ AbstractValue::Const(offset) => {
                            tmemory.insert(offset.clone(), value.clone());
                        }
                        _ => {}
                    }
                };
            }
            Opcode::PUSH0 => {
                push_or_break_val!(stack, AbstractValue::Const(U256::ZERO))
            }
            Opcode::DUP1 => {
                handle_dup!(stack, 1);
            }
            Opcode::DUP2 => {
                handle_dup!(stack, 2);
            }
            Opcode::DUP3 => {
                handle_dup!(stack, 3);
            }
            Opcode::DUP4 => {
                handle_dup!(stack, 4);
            }
            Opcode::DUP5 => {
                handle_dup!(stack, 5);
            }
            Opcode::DUP6 => {
                handle_dup!(stack, 6);
            }
            Opcode::DUP7 => {
                handle_dup!(stack, 7);
            }
            Opcode::DUP8 => {
                handle_dup!(stack, 8);
            }
            Opcode::DUP9 => {
                handle_dup!(stack, 9);
            }
            Opcode::DUP10 => {
                handle_dup!(stack, 10);
            }
            Opcode::DUP11 => {
                handle_dup!(stack, 11);
            }
            Opcode::DUP12 => {
                handle_dup!(stack, 12);
            }
            Opcode::DUP13 => {
                handle_dup!(stack, 13);
            }
            Opcode::DUP14 => {
                handle_dup!(stack, 14);
            }
            Opcode::DUP15 => {
                handle_dup!(stack, 15);
            }
            Opcode::DUP16 => {
                handle_dup!(stack, 16);
            }

            Opcode::LOG0 => {
                pop_or_break!(stack);
                pop_or_break!(stack);
            }
            Opcode::LOG1 => {
                pop_or_break!(stack);
                pop_or_break!(stack);

                pop_or_break!(stack);
            }
            Opcode::LOG2 => {
                pop_or_break!(stack);
                pop_or_break!(stack);

                pop_or_break!(stack);
                pop_or_break!(stack);
            }
            Opcode::LOG3 => {
                pop_or_break!(stack);
                pop_or_break!(stack);

                pop_or_break!(stack);
                pop_or_break!(stack);
                pop_or_break!(stack);
            }
            Opcode::LOG4 => {
                pop_or_break!(stack);
                pop_or_break!(stack);

                pop_or_break!(stack);
                pop_or_break!(stack);
                pop_or_break!(stack);
                pop_or_break!(stack);
            }
            Opcode::SWAP1 => {
                handle_swap!(stack, 1);
            }
            Opcode::SWAP2 => {
                handle_swap!(stack, 2);
            }
            Opcode::SWAP3 => {
                handle_swap!(stack, 3);
            }
            Opcode::SWAP4 => {
                handle_swap!(stack, 4);
            }
            Opcode::SWAP5 => {
                handle_swap!(stack, 5);
            }
            Opcode::SWAP6 => {
                handle_swap!(stack, 6);
            }
            Opcode::SWAP7 => {
                handle_swap!(stack, 7);
            }
            Opcode::SWAP8 => {
                handle_swap!(stack, 8);
            }
            Opcode::SWAP9 => {
                handle_swap!(stack, 9);
            }
            Opcode::SWAP10 => {
                handle_swap!(stack, 10);
            }
            Opcode::SWAP11 => {
                handle_swap!(stack, 11);
            }
            Opcode::SWAP12 => {
                handle_swap!(stack, 12);
            }
            Opcode::SWAP13 => {
                handle_swap!(stack, 13);
            }
            Opcode::SWAP14 => {
                handle_swap!(stack, 14);
            }
            Opcode::SWAP15 => {
                handle_swap!(stack, 15);
            }
            Opcode::SWAP16 => {
                handle_swap!(stack, 16);
            }
            Opcode::JUMPDEST => {}
            op => {
                if is_push(op) {
                    let v = U256::from_be_slice(ins.input.as_slice());
                    push_or_break_val!(stack, AbstractValue::Const(v));
                } else {
                    panic!("Abstract opcode not implemented {:?}", op);
                }
            }
        }
        self.pc += 1;

        return StepResult::Ok;
    }
}

struct Analysis {
    storage_slots: Vec<AbstractValueRef>,
    external_contracts: Vec<AbstractValueRef>,
    slot_mask_size: HashMap<U256, usize>,
}
impl Analysis {
    fn new() -> Self {
        Self {
            storage_slots: Vec::new(),
            external_contracts: Vec::new(),
            slot_mask_size: HashMap::new(),
        }
    }
}
struct AbstractVM {
    analysis: Analysis,
    stack: Vec<AbstractVMInstance>,
}

impl AbstractVM {
    fn new(
        program_bytes: Rc<Vec<u8>>,
        program: Rc<Vec<Operation>>,
        jump_dests: Rc<HashMap<usize, usize>>,
        start: usize,
    ) -> Self {
        let current = AbstractVMInstance::new(
            program_bytes.clone(),
            program.clone(),
            jump_dests.clone(),
            start,
        );

        Self {
            analysis: Analysis::new(),
            stack: vec![current],
        }
    }

    pub fn run(&mut self) {
        let mut visited = HashSet::new();
        visited.insert(self.stack.last().unwrap().pc);

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
                let res = vm.step(self.analysis.borrow_mut());

                match res {
                    StepResult::Stop => {
                        break;
                    }
                    StepResult::Split(pc) => {
                        if visited.contains(&pc) {
                            continue;
                        }
                        visited.insert(pc);
                        self.stack.push(vm.copy(pc))
                    }
                    _ => {}
                }
            }
        }
    }
}

fn convert_to_address(v: &HeapRef<AbstractValue>) -> Option<Address> {
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
    do_abstract_analysis: bool,
) -> eyre::Result<(Vec<AnalyzedStoragesSlot>, Vec<Address>)> {
    let start = std::time::Instant::now();

    let bytes: Vec<u8> = bytecode.bytes().to_vec();
    let contract_instructions = disassemble_bytes(bytes.clone());

    let add_slot = move |v: &mut Vec<AnalyzedStoragesSlot>, slot: &[u8]| {
        let slot = U256::from_be_slice(slot);
        v.push(AnalyzedStoragesSlot::Slot(slot, SlotType::Uint));
    };

    if contract_instructions.len() <= 3 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut slots = Vec::<AnalyzedStoragesSlot>::with_capacity(64);

    let mut jump_dests = HashMap::<usize, usize>::new();

    for i in 0..contract_instructions.len() {
        let ins0 = &contract_instructions[i];
        if ins0.opcode == Opcode::JUMPDEST {
            jump_dests.insert(ins0.offset as usize, i);
        }
    }
    for i in 0..contract_instructions.len() - 3 {
        let ins0 = &contract_instructions[i];
        let ins1 = &contract_instructions[i + 1];
        let ins2 = &contract_instructions[i + 2];

        if ins1.opcode == Opcode::PUSH0 {
            match ins2.opcode {
                Opcode::SSTORE | Opcode::SLOAD => {
                    slots.push(AnalyzedStoragesSlot::Slot(U256::ZERO, SlotType::Uint));
                }
                _ => {}
            }
        }
        if is_push(ins1.opcode) {
            match ins2.opcode {
                Opcode::SSTORE | Opcode::SLOAD => {
                    add_slot(&mut slots, ins1.input.as_slice());
                }
                _ => {}
            }
        }
        if is_push(ins0.opcode) && ins1.opcode == Opcode::DUP1 {
            match ins2.opcode {
                Opcode::SSTORE | Opcode::SLOAD => {
                    add_slot(&mut slots, ins0.input.as_slice());
                }
                _ => {}
            }
        }
    }

    let result = if do_abstract_analysis {
        let (external_refs, storage_slots) = {
            let program = Rc::new(contract_instructions);
            let program_bytes = Rc::new(bytes);
            let jump_dests = Rc::new(jump_dests);
            let mut vm = AbstractVM::new(
                program_bytes.clone(),
                program.clone(),
                jump_dests.clone(),
                0,
            );
            vm.run();

            (vm.analysis.external_contracts, vm.analysis.storage_slots)
        };

        let storage_slots = storage_slots
            .into_iter()
            .map(|v| reduce(&v))
            .collect::<Vec<_>>();

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

        let out_slots = storage_slots
            .into_iter()
            .map(|v| convert(&v))
            .flatten()
            .unique()
            .collect::<Vec<AnalyzedStoragesSlot>>();

        let cont_refs = external_refs
            .into_iter()
            .map(|v| convert(&v))
            .flatten()
            .unique()
            .collect::<Vec<AnalyzedStoragesSlot>>();
        let mut result: HashMap<U256, AnalyzedStoragesSlot> = HashMap::new();
        for slot in slots.iter().chain(out_slots.iter()).chain(cont_refs.iter()) {
            match result.entry(slot.get_slot()) {
                Entry::Occupied(mut entry) => {
                    entry.insert(entry.get().promote(slot));
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

        (out_slots, contract_refs)
    } else {
        (slots, vec![])
    };

    log::debug!(target: LOGGER_TARGET_MAIN, "analysis took {:?}", start.elapsed());
    return Ok(result);
}

#[cfg(test)]
mod tests {
    use alloy::hex;
    use alloy_provider::Provider;
    use revm::primitives::Bytecode;

    use crate::utils::provider_from_string;

    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn it_works() {
        // let test_addr = Address::from(hex!("bf1c0206de440b2cf76ea4405e1dbf2fc227a463"));
        let test_addr = Address::from(hex!("784955641292b0014bc9ef82321300f0b6c7e36d"));
        // let test_addr = Address::from(hex!("ac3E018457B222d93114458476f3E3416Abbe38F"));

        // let test_addr = Address::from(hex!("7effd7b47bfd17e52fb7559d3f924201b9dbff3d"));
        // let test_addr = Address::from(hex!("BBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb"));
        // let test_addr: Address = Address::from(hex!("5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"));
        let url = std::env::var_os("PROVIDER").unwrap();
        let provider = provider_from_string(&url.to_string_lossy().to_string())
            .await
            .unwrap();

        let code = Bytecode::new_raw(provider.get_code_at(test_addr).await.unwrap());

        match perform_analysis(&code, true) {
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
