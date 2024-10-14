use evm_disassembler::{Opcode, Operation};
use itertools::Itertools;
use revm::primitives::{Address, Bytecode, U256};

use crate::LOGGER_TARGET_MAIN;

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

pub fn perform_analysis(bytecode: &Bytecode) -> eyre::Result<(Vec<Address>, Vec<U256>)> {
    let start = std::time::Instant::now();
    let bytes: Vec<u8> = bytecode.bytes().to_vec();
    let contract_instructions = disassemble_bytes(bytes.clone());

    let add_slot = move |v: &mut Vec<U256>, slot: &[u8]| {
        let slot = U256::from_be_slice(slot);
        v.push(slot);
    };

    if contract_instructions.len() <= 3 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut addresses = Vec::<Address>::with_capacity(32);
    let mut slots = Vec::<U256>::with_capacity(64);

    for i in 0..contract_instructions.len() - 3 {
        let ins0 = &contract_instructions[i];
        let ins1 = &contract_instructions[i + 1];
        let ins2 = &contract_instructions[i + 2];

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

        match (ins0.opcode, ins1.opcode, ins2.opcode) {
            (_, Opcode::PUSH0, Opcode::SSTORE) => {
                slots.push(U256::from(0));
            }
            (Opcode::PUSH32, Opcode::PUSH20, Opcode::AND) => {
                let address = Address::from_slice(&ins0.input[12..]);
                addresses.push(address);
            }
            _ => {}
        }
    }

    log::info!(target: LOGGER_TARGET_MAIN, "Disassembly took {:?}", start.elapsed());
    Ok((
        addresses.into_iter().unique().collect::<Vec<_>>(),
        slots.into_iter().unique().collect::<Vec<_>>(),
    ))
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
        let test_addr = Address::from(hex!("7effd7b47bfd17e52fb7559d3f924201b9dbff3d"));
        let url = std::env::var_os("PROVIDER").unwrap();
        let provider = provider_from_string(&url.to_string_lossy().to_string())
            .await
            .unwrap();

        let code = Bytecode::new_raw(provider.get_code_at(test_addr).await.unwrap());

        match perform_analysis(&code) {
            Ok(data) => {
                println!("Analyzed");
                println!("Addresses: {:?}", data.0);
                println!("Slots: {:?}", data.1);
            }
            Err(e) => {
                println!("Failed to analyze {}", e);
            }
        };
    }
}
