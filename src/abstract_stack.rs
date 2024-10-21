use revm::interpreter::InstructionResult;

use crate::{
    abstract_value::{AbstractValue, AbstractValueRef, HeapRef},
    analysis::LOGGER_TARGET_ANALYSIS,
    LOGGER_TARGET_LOADS,
};

use match_deref::match_deref;

pub const STACK_LIMIT: usize = 1024;

#[derive(Debug)]
pub(crate) struct AbstractStack {
    pub(crate) data: Vec<AbstractValueRef>,
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
    pub fn pop(&mut self) -> Result<AbstractValueRef, InstructionResult> {
        match self.data.pop() {
            Some(v) => Ok(v.clone()),
            None => {
                println!("pop: stack underflow");
                Err(InstructionResult::StackUnderflow)
            }
        }
    }

    #[inline]
    pub fn drop_n(&mut self, n: usize) -> Result<(), InstructionResult> {
        for _ in 0..n {
            self.pop()?;
        }
        Ok(())
    }

    /// Push a new value onto the stack.
    ///
    /// If it will exceed the stack limit, returns `StackOverflow` error and leaves the stack
    /// unchanged.
    #[inline]
    pub fn push(&mut self, value: AbstractValueRef) -> Result<(), InstructionResult> {
        assume!(self.data.capacity() == STACK_LIMIT);
        let value = match_deref! {
            match &value {
                Deref @ AbstractValue::TypeMask(t0, Deref @ AbstractValue::TypeMask(t1, inner)) => HeapRef::new(AbstractValue::TypeMask(t0.promote(t1), inner.clone())),
                _ => value.clone()
            }
        };

        if self.data.len() == STACK_LIMIT {
            log::trace!(target: LOGGER_TARGET_ANALYSIS, "push: stack overflow");
            return Err(InstructionResult::StackOverflow);
        }
        self.data.push(value);
        Ok(())
    }

    #[inline]
    pub fn push_uint(&mut self, value: u64) -> Result<(), InstructionResult> {
        self.push(AbstractValue::u64(value))
    }

    #[inline]
    pub fn dup(&mut self, n: usize) -> Result<(), InstructionResult> {
        assume!(n > 0, "attempted to dup 0");
        if n > self.data.len() {
            log::trace!(target: LOGGER_TARGET_ANALYSIS, "dup: stack underflow");
            return Err(InstructionResult::StackUnderflow);
        }
        let element = self.data.get(self.data.len() - n);
        match element {
            None => {
                log::trace!(target: LOGGER_TARGET_ANALYSIS, "dup: stack overflow");
                Err(InstructionResult::StackOverflow)
            }
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
            log::trace!(target: LOGGER_TARGET_ANALYSIS, "swap: stack overflow");
            return Err(InstructionResult::StackOverflow);
        }
        self.data.swap(top - 1, top - nth - 1);
        Ok(())
    }
}
