import { createSimulator } from "./index.cjs";

const mainProvider = process.env.PROVIDER!;

const run = async () => {
  let simulator = await createSimulator(mainProvider, "Reth");

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
      data: "0x",
    },
    (logData) => {
      // Called if any logs are emitted during the execution
    }
  );

  console.log("Alice balance:", await fork.getBalance(alice));
  console.log("Bob balance:", await fork.getBalance(bob));

  console.log(
    "Execution result:",
    executionResult.receipt.status === 1 // true if the transaction was successful
  );

  await simulator.preload([
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "0xdac17f958d2ee523a2206206994597c13d831ec7",
  ]);
};

run();
