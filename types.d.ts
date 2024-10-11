
export type LinkType = 'Reth' | 'Geth';
export interface Config {
    linkType: LinkType;
    forkUrl: string;
    wsForkUrl: string;
    secondsPerBlock: bigint;
}
export interface ExecutionResult {
    exceptionError?: {
        error: string;
        errorType: string;
    };
    returnValue: string;
}

export type AccessList = Array<[account: string, reads: Array<string>]>;

export interface Log {
    address: string;
    topics: string[];
    data: string;
}
interface SimReceipt {
    logs: Log[];
    cumulativeGasUsed: bigint;
    gasUsed: bigint;
    status: number;
}

export interface SimResult {
    gasUsed: bigint;
    receipt: SimReceipt;
    execResult: ExecutionResult;
    accessList: AccessList;
}

interface SimTransactionRequest {
    from: string;
    to: string;
    data: string;
    value: bigint;
}

export type OnLogFn = (log: Log) => void;

export interface SimulatorFork {
    commitTx: (tx: SimTransactionRequest, step: OnLogFn) => Promise<SimResult>;
    simulateTx: (tx: SimTransactionRequest, step: OnLogFn) => Promise<SimResult>;
    checkpoint: () => Promise<number>;
    revertTo: (id: number) => Promise<void>;
    mine: (toMine: number) => Promise<void>;
    getBlockNumber: () => Promise<number>;
    getTimestamp: () => Promise<number>;
    setAccountCode: (address: string, code: string) => Promise<void>;
    setTimestamp: (timestamp:  number) => Promise<void>;
    setBalance: (address: string, balance: bigint) => Promise<void>;
    setContractStorage: (address: string, key: bigint, value: bigint) => Promise<void>;
    getStorageAt: (address: string, key: bigint) => Promise<bigint>;
    getBalance: (address: string) => Promise<bigint>;
    preload: (address: string) => void;
}

export interface ForkySimulator {
    onBlock: (blockNumber: number) => Promise<void>;
    fork: () => Promise<SimulatorFork>;
    preload: (address: string) => void;
}

/**
 * 
 * @param stateProvider Provider used for getBlock, storageAt, getCode, getBalance, getNonce
 * @param traceProvider Provider used for to block traces. 'debug_trace_block_by_number' for Geth and 'trace_replay_block_transactions' for Reth
 * @param linkType LinkType.Reth or LinkType.Geth
 * @param opts Options for the simulator
 * @param opts.secondsPerBlock How many seconds to add to the timestamp when 'mine' is called on a fork. Defaults to 12
 * @param opts.forkBlock Block to fork from (optional), defaults to the current block. Defaults to the current block
 * @param opts.traceProvider Provider used for to block traces. 'debug_trace_block_by_number' for Geth and 'trace_replay_block_transactions' for Reth defaults to 'stateProvider'
 * @param opts.maxBlocksBehind Sets the large number of blocks we want to sync at once before we reset the state.
 * This is mostly to prevent cases during development where a simulator process wakes up after a long time and traces a large number of blocks.
 * Defaults to 50
 */
export function createSimulator(
    stateProvider: string,
    linkType: LinkType,
    opts?: {
        secondsPerBlock?: number,
        forkBlock?: number
        traceProvider?: string,
        maxBlocksBehind?: number
    }
): Promise<ForkySimulator>;
