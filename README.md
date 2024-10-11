# Forky - A EVM with state synchronization capabilities

Forky was made with the intention making it easy and quick to fork and simulate transactions against the tip of the chain. Forky maintains an internal cannonical replica of the chain it is following.

This replica starts empty, but as the application is doing forks and executing transactions it will work as a read through cache and populate the replicated state with the live state. As the chain produces blocks Forky will then query state updates from upstream and apply them to any previously requested state.

This means that Forky only ever requests a piece of state from upstream once, and will always use the most fresh state for simulations. This is very usefull for applications that do a lot of short lived forks and simulations against the tip of the chain, something that is very common in backends that service DeFi applications.

MEV users may also find it useful, but it depends on the application. As Forky works best if the transactions that need to be simulated are interacting with a mostly constrained set of contracts.

## Setup

Forky uses these main endpoints for pulling state from upstream:

- `eth_getBlockByNumber`
- `eth_getBalance`
- `eth_getCode`
- `eth_getTransactionCount`
- `eth_getStorageAt`

Forky requires that your rpc provider serves one of the following endpoints for state synchronization:

- `replay_block_transactions` (for Reth nodes)
- `debug_trace_block_by_number` (for Geth nodes)


## Usage

```typescript
import { createSimulator } from "@slot0/forky";

const mainProvider = "ws://..." // or http://...
const provider = new WebSocketProvider(mainProvider);

// (Optional) only if you want to use a different provider for tracing
const traceProvider = "ws://..." // or http://...

let simulator = await createSimulator(
    mainProvider,
    LinkType.Reth,
    {
        traceProvider,
        secondsPerBlock: 12,
        // forkBlock: ... explicit fork block for starting at specific historical blocks
        // maxBlocksBehind: 50 ... how many blocks to sync at once before resetting the state, may need to be adjusted for networks with fast block times
    }
);

// Forky only synchronizes state when onBlock is called. So this needs to be called by
// the application whenever a new block is created.

// onBlock will throw an error if synchronization failed.
// This will invalidate the current simulator and requires us to create a new one.
// If the blockNumber delta is too large the simulator may purge the state (see maxBlocksBehind)
provider.on("block", async blockNumber => {
    try {
        await simulator.onBlock(blockNumber)
    } catch(e) {
        console.error(e);
        // Simulator desynchronized, create new one and set up onBlock again
        simulator = await createSimulator(
            mainProvider,
            traceProvider,
            LinkType.Reth,
            12,
        );
        setupOnBlock();
    }
});


// Create new fork to simulate on
const fork = await simulator.fork();

const alice = "0xac1872e0701E1DF6c302Fad31902dd67DA121B8E";
const bob = "0x850FC93BA787E49C795b86BD0feD4d87d7342518";
console.log("Alice balance:", await fork.getBalance(alice));
console.log("Bob balance:", await fork.getBalance(bob));

// give ourselves some ether
await fork.setBalance(alice, 100n * 10n ** 18n);
console.log("Alice balance:", await fork.getBalance(alice));

// Send some ether from alice to bob
const executionResult = await fork.commitTx(
    {
        from: alice,
        to: bob,
        value: 10n * 10n ** 18n,
        data: "0x"
    },
    logData => {
        // Called if any logs are emitted during the execution
    }
)
console.log("Alice balance:", await fork.getBalance(alice));
console.log("Bob balance:", await fork.getBalance(bob));

console.log(
    "Execution result:",
    executionResult.receipt.status === 1 // true if the transaction was successful
);

// Should output:
// Alice balance: 0n
// Bob balance: 0n
// Alice balance: 100000000000000000000n
// Alice balance: 89999999999999979000n
// Bob balance: 10000000000000000000n
// Execution result: true

```

## Building

Building requires a [supported version of Node and Rust](https://github.com/neon-bindings/neon#platform-support).

To run the build, run:

```sh
$ npm run build
```

This command uses the [@neon-rs/cli](https://www.npmjs.com/package/@neon-rs/cli) utility to assemble the binary Node addon from the output of `cargo`.


#### `npm install`

Installs the project, including running `npm run build`.

#### `npm run build`

Builds the Node addon (`index.node`) from source, generating a release build with `cargo --release`.

Additional [`cargo build`](https://doc.rust-lang.org/cargo/commands/cargo-build.html) arguments may be passed to `npm run build` and similar commands. For example, to enable a [cargo feature](https://doc.rust-lang.org/cargo/reference/features.html):

```
npm run build -- --feature=beetle
```

#### `npm run debug`

Similar to `npm run build` but generates a debug build with `cargo`.

#### `npm run cross`

Similar to `npm run build` but uses [cross-rs](https://github.com/cross-rs/cross) to cross-compile for another platform. Use the [`CARGO_BUILD_TARGET`](https://doc.rust-lang.org/cargo/reference/config.html#buildtarget) environment variable to select the build target.
