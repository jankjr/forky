use crate::opcodes::{Opcode, Operation};
use alloy::{hex, primitives::U160, rpc::types::Block};
use itertools::Itertools;
use match_deref::match_deref;
use revm::{
    interpreter::{instructions::i256::i256_cmp, InstructionResult},
    primitives::{keccak256, Address, BlockEnv, Bytecode, FixedBytes, I256, U256},
};

use std::{
    borrow::BorrowMut,
    collections::{hash_map::Entry, HashMap, HashSet},
    fmt::{Debug, Display},
    hash::Hash,
    rc::Rc,
};

pub type HeapRef<T> = Rc<T>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum SlotType {
    Address,
    Unknown,
    String,
    Byte,
    Bool,
    Bytes(usize),
    Tuple(usize),
}

impl Display for SlotType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl SlotType {
    fn order(&self) -> usize {
        match self {
            SlotType::Bool => 1000,
            SlotType::String => 2000,
            SlotType::Unknown => 0,
            SlotType::Address => 20,
            SlotType::Byte => 1,
            SlotType::Bytes(v) => *v,
            SlotType::Tuple(fields) => 1000 + *fields * 32,
        }
    }
    pub fn promote(&self, other: &SlotType) -> SlotType {
        if self.order() >= other.order() {
            return self.clone();
        }
        return other.clone();
    }
}

#[derive(Debug, Clone, Eq, Copy, PartialEq, PartialOrd, Ord, Hash)]
pub enum ArgType {
    Calldata(u32, usize),
    MsgSender,
    This,
}

impl Display for ArgType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArgType::Calldata(selector, index) => {
                write!(f, "Calldata({selector:#08x}, arg={index})")
            }
            ArgType::MsgSender => write!(f, "MsgSender"),
            ArgType::This => write!(f, "This"),
        }
    }
}

impl ArgType {
    pub fn get_type(&self) -> SlotType {
        match self {
            ArgType::Calldata(_, _) => SlotType::Unknown,
            ArgType::MsgSender => SlotType::Address,
            ArgType::This => SlotType::Address,
        }
    }
    pub fn selector(&self) -> u32 {
        match self {
            ArgType::Calldata(selector, _) => *selector,
            _ => 0u32,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbstractValue {
    Calldata(usize),
    CalldataArray(usize, usize),
    Const(U256),
    AddressConst(Address),
    OpcodeResult(Opcode),
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
    TypeMask(SlotType, HeapRef<AbstractValue>),
}

pub type AbstractValueRef = HeapRef<AbstractValue>;

pub static ADDR_MASK: U256 = U256::from_be_slice(&hex!("ffffffffffffffffffffffffffffffffffffffff"));
static BYTE_MASK: U256 = U256::from_be_slice(&hex!("ff"));

impl Display for AbstractValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbstractValue::TypeMask(t, inner) => write!(f, "{}({})", t, inner.to_string()),
            AbstractValue::AddressConst(inner) => write!(f, "{}", inner.to_string()),
            AbstractValue::Const(val) => write!(f, "{}", hex::encode(val.to_be_bytes::<32>())),
            AbstractValue::Mapping(arg_t, slot) => {
                write!(f, "mapping({:?} => {})", arg_t, slot.to_string())
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
                    None => format!("UNKNOWN"),
                };

                match value {
                    Some(value) => write!(
                        f,
                        "{:?}{}.{}[value:{:?}]({})",
                        op,
                        sel,
                        address.to_string(),
                        value,
                        args.iter().map(|v| v.to_string()).join(", ")
                    ),
                    None => write!(
                        f,
                        "{:?}{}.{}({})",
                        op,
                        sel,
                        address.to_string(),
                        args.iter().map(|v| v.to_string()).join(", ")
                    ),
                }
            }
            _ => write!(f, "{:?}", self),
        }
    }
}
impl AbstractValue {
    pub fn arg_source(&self, sel: u32) -> Vec<ArgType> {
        match self {
            AbstractValue::TypeMask(_, inner) => inner.arg_source(sel),
            AbstractValue::Calldata(offset) => vec![ArgType::Calldata(sel, *offset)],
            AbstractValue::OpcodeResult(Opcode::CALLER) => vec![ArgType::MsgSender],
            AbstractValue::OpcodeResult(Opcode::ADDRESS) => vec![ArgType::This],
            AbstractValue::CalldataArray(offset, _) => vec![ArgType::Calldata(sel, *offset)],
            AbstractValue::BinOpResult(_, a, b) => a
                .arg_source(sel)
                .iter()
                .chain(b.arg_source(sel).iter())
                .cloned()
                .collect::<Vec<_>>(),
            AbstractValue::TertiaryOpResult(_, a, b, c) => a
                .arg_source(sel)
                .iter()
                .chain(b.arg_source(sel).iter())
                .chain(c.arg_source(sel).iter())
                .cloned()
                .collect::<Vec<_>>(),
            AbstractValue::UnaryOpResult(_, a) => a.arg_source(sel),
            _ => Vec::new(),
        }
    }

    pub fn slot(&self) -> Option<U256> {
        match self {
            AbstractValue::TypeMask(_, inner) => inner.slot(),
            AbstractValue::StorageSlot(_, slot) => Some(*slot),
            AbstractValue::StorageArray(inner) => inner.slot(),
            AbstractValue::Mapping(_, slot) => slot.slot(),
            AbstractValue::UnaryOpResult(_, inner) => inner.slot(),
            _ => None,
        }
    }
    pub fn bin(op: Opcode, a: AbstractValueRef, b: AbstractValueRef) -> AbstractValueRef {
        use AbstractValue::*;
        use Opcode::*;
        match_deref! {
            match (&a, &b) {
                (Deref@TypeMask(ta, inner_a), Deref@TypeMask(tb, inner_b)) => {
                    return HeapRef::new(TypeMask(ta.promote(tb), AbstractValue::bin(op, inner_a.clone(), inner_b.clone())));
                }
                (Deref@TypeMask(ta, inner_a), inner_b) => {
                    if *ta == SlotType::String {
                        return HeapRef::new(TypeMask(*ta, AbstractValue::bin(op, inner_a.clone(), inner_b.clone())));
                    }
                }
                (inner_a, Deref@TypeMask(tb, inner_b)) => {
                    if *tb == SlotType::String {
                        return HeapRef::new(TypeMask(*tb, AbstractValue::bin(op, inner_a.clone(), inner_b.clone())));
                    }
                }
                _ => ()
            }
        }

        match_deref! {
            match (op, &a, &b) {
                (ADD, Deref@Const(a), Deref@Const(b)) => {
                    return AbstractValue::val(&a.wrapping_add(*b));
                }
                (SHA3, Deref @ Const(a), Deref @ Const(b)) => {
                    let mut buff = [0u8; 64];
                    buff[0..32].copy_from_slice(&a.to_be_bytes::<32>());
                    buff[32..64].copy_from_slice(&b.to_be_bytes::<32>());

                    let index_bytes = keccak256(&buff);
                    let storage_slot: U256 = index_bytes.into();
                    return AbstractValue::val(&storage_slot);
                }


                (MUL, Deref@Const(a), Deref@Const(b)) => {
                    return AbstractValue::val(&a.wrapping_mul(*b));
                }
                (AND, Deref@Const(a), Deref@Const(b)) => {
                    return AbstractValue::val(&(a & b));
                }
                (OR, Deref@Const(a), Deref@Const(b)) => {
                    return AbstractValue::val(&(a | b));
                }
                (XOR, Deref@Const(a), Deref@Const(b)) => {
                    return AbstractValue::val(&(a ^ b));
                }
                (MUL|AND|OR|XOR, Deref@Const(_), Deref@Const(_)) => {
                    return HeapRef::new(BinOpResult(op, a.clone(), b.clone()))
                }
                // Move constants to the left on commutative operations
                (ADD|MUL|AND|OR|XOR, exp, Deref@Const(v)) => {
                    return AbstractValue::bin(op, AbstractValue::val(v), exp.clone());
                }
                _ => {}
            }
        };
        // Fold constants on associate operators
        match_deref! {
            match (op, &a, &b) {
                (op, Deref@Const(a), Deref@BinOpResult(op1, Deref@Const(b), exp)) => {
                    if op == *op1 && (op == ADD || op == MUL || op == AND || op == OR || op == XOR) {
                        return AbstractValue::bin(op, AbstractValue::bin(op, AbstractValue::val(a), AbstractValue::val(b)), exp.clone());
                    }
                },
                _ => {}
            }
        };

        // Identities and other stuff
        match_deref! {
            match (op, &a, &b) {
                (DIV, exp, Deref@Const(bv)) => {
                    if bv.eq(&U256::from(1)) {
                        return exp.clone();
                    }
                    // if bv.count_ones() == 1 {
                    //     // It is a right shift

                    //     let size = (256 - bv.leading_zeros()) / 8;

                    //     return HeapRef::new(AbstractValue::TypeMask(SlotType::Bytes(size), exp.clone()))
                    // }
                }

                (AND, Deref@Const(v), Deref@TypeMask(_, inner)) => {
                    return AbstractValue::bin(AND, AbstractValue::val(v), inner.clone());
                }
                (SUB, Deref @ BinOpResult(ADD, Deref@Const(a), exp), Deref@Const(b)) => {
                    if let (Ok(a), Ok(b)) = ((*a).try_into(), (*b).try_into()) as (Result<I256,_>, Result<I256,_>) {
                        let res = a.wrapping_sub(b);
                        if res.is_zero() {
                            return exp.clone();
                        }
                        if !res.is_negative() {
                            if let Ok(res) = res.try_into() as Result<U256,_> {
                                return AbstractValue::bin(ADD, AbstractValue::val(&res), exp.clone());
                            }
                        }


                    };


                    // if a == b {
                    //     return AbstractValue::val(&U256::ZERO);
                    // }
                }
                (AND, Deref@Const(v), inner) => {
                    if v.is_zero() {
                        return AbstractValue::val(&U256::ZERO);
                    }

                    if v.eq(&U256::from(1)) {
                        return HeapRef::new(AbstractValue::TypeMask(SlotType::String, inner.clone()))
                    }
                    if v.eq(&ADDR_MASK) {
                        return HeapRef::new(AbstractValue::TypeMask(SlotType::Address, inner.clone()))
                    }
                    if v.eq(&BYTE_MASK) {
                        return HeapRef::new(AbstractValue::TypeMask(SlotType::Byte, inner.clone()))
                    }
                    if (v + U256::from(1)).is_power_of_two() {
                        let ones = v.count_ones();
                        let bytes = (ones + 7) / 8;
                        return HeapRef::new(AbstractValue::TypeMask(SlotType::Bytes(bytes), inner.clone()))
                    }
                }
                (MUL, Deref@Const(va), exp) => {
                    if va.eq(&U256::from(1)) {
                        return exp.clone();
                    }
                }
                // (ADD, Deref@Const(val), Deref@Calldata(offset)) => {
                //     return HeapRef::new(AbstractValue::CalldataArray(*offset, val.to()));
                // }
                _ => {}
            }
        };

        HeapRef::new(BinOpResult(op, a.clone(), b.clone()))
    }
    pub fn map(input: SlotType, a: AbstractValueRef) -> AbstractValueRef {
        HeapRef::new(AbstractValue::Mapping(input, a.clone()))
    }
    pub fn array(a: AbstractValueRef) -> AbstractValueRef {
        HeapRef::new(AbstractValue::StorageArray(a.clone()))
    }
    pub fn unary(op: Opcode, a: AbstractValueRef) -> AbstractValueRef {
        use AbstractValue::*;
        use Opcode::*;
        match_deref! {
            match (op, &a) {
                (ISZERO, Deref @ UnaryOpResult(ISZERO, Deref @ UnaryOpResult(ISZERO, _))) => {
                    return HeapRef::new(TypeMask(SlotType::Bool, HeapRef::new(UnaryOpResult(ISZERO, a.clone()))));
                }
                (_, Deref @ TypeMask(t, inner)) => {
                    return HeapRef::new(TypeMask(*t, AbstractValue::unary(op, inner.clone())));
                }
                (_, Deref @ TypeMask(_, Deref @ TypeMask(tinner, inner))) => {
                    return HeapRef::new(TypeMask(*tinner, inner.clone()))
                }
                _ => {}
            }
        };
        HeapRef::new(AbstractValue::UnaryOpResult(op, a.clone()))
    }
    pub fn op(op: Opcode) -> AbstractValueRef {
        HeapRef::new(AbstractValue::OpcodeResult(op))
    }
    pub fn val(a: &U256) -> AbstractValueRef {
        HeapRef::new(AbstractValue::Const(*a))
    }
    pub fn u64(a: u64) -> AbstractValueRef {
        HeapRef::new(AbstractValue::Const(U256::from(a)))
    }

    pub fn ext_call(
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

    pub fn static_or_delegate_call(
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

pub fn infer_type(v: &HeapRef<AbstractValue>) -> SlotType {
    match **v {
        AbstractValue::TypeMask(t, _) => t,
        _ => v
            .arg_source(0)
            .first()
            .map(|v| v.get_type())
            .unwrap_or(SlotType::Unknown),
    }
}

pub fn reduce_externals(v: &HeapRef<AbstractValue>) -> HeapRef<AbstractValue> {
    use AbstractValue::*;
    HeapRef::new(match_deref! {
        match v {
            Deref @ AddressConst(_) => return v.clone(),
            Deref @ Const(v) => return HeapRef::new(AddressConst(Address::from(U160::from(*v)))),
            Deref @ TypeMask(SlotType::Address, Deref@Const(v)) => return HeapRef::new(AddressConst(Address::from(U160::from(*v)))),
            Deref @ TypeMask(SlotType::Address, Deref@StorageSlot(SlotType::Address, s)) => AddressConst(Address::from(U160::from(*s))),
            Deref @ TypeMask(SlotType::Bytes(20), Deref@StorageSlot(SlotType::Bytes(20), s)) => AddressConst(Address::from(U160::from(*s))),
            v => return reduce(v, &vec![])
        }
    })
}

pub fn infer_type_from_offset(offset: &U256) -> SlotType {
    if offset.is_zero() {
        SlotType::Unknown
    } else {
        SlotType::Tuple(offset.to::<usize>() + 1usize)
    }
}

pub fn reduce(
    original: &HeapRef<AbstractValue>,
    arg_sources: &Vec<ArgType>,
) -> HeapRef<AbstractValue> {
    use crate::abstract_value::AbstractValue::*;
    use Opcode::*;

    let sload_exp = match_deref! {
        match original {
            Deref @ StorageSlot(_, _) => return original.clone(),
            Deref @ Mapping(_, _) => return original.clone(),
            Deref @ StorageArray(_) => return original.clone(),
            Deref @ TypeMask(t, inner) => return HeapRef::new(TypeMask(t.clone(), reduce(&inner, arg_sources))),
            Deref @ UnaryOpResult(SLOAD, exp) => exp,
            _ => {
                return original.clone();
            }
        }
    };

    match_deref! {
        match sload_exp {
            Deref @ BinOpResult(ADD, Deref@Const(offset), Deref@UnaryOpResult(SHA3, Deref@Const(slot))) => return AbstractValue::array(HeapRef::new(StorageSlot(infer_type_from_offset(offset), *slot))),
            Deref @ BinOpResult(ADD, Deref@Const(offset), Deref@BinOpResult(SHA3, arg1, Deref @ BinOpResult(SHA3, arg0, Deref @ Const(v)))) => {
                let arg_type0 = arg_sources.get(0).map(|v|v.get_type()).unwrap_or_else(||infer_type(arg0));
                let arg_type1 = arg_sources.get(1).map(|v|v.get_type()).unwrap_or_else(||infer_type(arg1));

                return AbstractValue::map(arg_type0, AbstractValue::map(arg_type1, HeapRef::new(StorageSlot(infer_type_from_offset(offset), *v))))
            },
            Deref @ Const(v) => return HeapRef::new(StorageSlot(SlotType::Unknown, *v)),
            Deref @ UnaryOpResult(SHA3, Deref@Const(v)) => return HeapRef::new(StorageArray(HeapRef::new(StorageSlot(SlotType::Unknown, *v)))),
            Deref @ BinOpResult(SHA3, arg1, Deref @ BinOpResult(SHA3, arg0, Deref @ Const(v))) => return AbstractValue::map(arg_sources.get(0).map(|v|v.get_type()).unwrap_or_else(||infer_type(arg0)), AbstractValue::map(arg_sources.get(1).map(|v|v.get_type()).unwrap_or_else(||infer_type(arg1)), HeapRef::new(StorageSlot(SlotType::Unknown, *v)))),
            Deref @ BinOpResult(SHA3, arg0, Deref @ Const(v)) => return AbstractValue::map(arg_sources.get(0).map(|v|v.get_type()).unwrap_or_else(||infer_type(arg0)), HeapRef::new(StorageSlot(SlotType::Unknown, *v))),
            Deref @ BinOpResult(SHA3, Deref @ Const(v), arg0) => return AbstractValue::map(arg_sources.get(0).map(|v|v.get_type()).unwrap_or_else(||infer_type(arg0)), HeapRef::new(StorageSlot(SlotType::Unknown, *v))),
            Deref @ BinOpResult(ADD, Deref@Const(offset), Deref@BinOpResult(SHA3, arg0, Deref@Const(slot_index))) => return AbstractValue::map(arg_sources.get(0).map(|v|v.get_type()).unwrap_or_else(||infer_type(arg0)), HeapRef::new(StorageSlot(infer_type_from_offset(offset), *slot_index))),
            Deref @ BinOpResult(ADD, Deref@Const(offset), Deref@BinOpResult(SHA3, Deref@Const(slot_index), arg0)) => return AbstractValue::map(arg_sources.get(0).map(|v|v.get_type()).unwrap_or_else(||infer_type(arg0)), HeapRef::new(StorageSlot(infer_type_from_offset(offset), *slot_index))),
            Deref @ BinOpResult(ADD, Deref@Const(_), Deref@UnaryOpResult(SHA3, exp)) => return AbstractValue::array(reduce(exp, arg_sources)),

            _ => {
                // println!("Failed to reduce {:?}", original);
                return original.clone();
            }
        }
    }
}
