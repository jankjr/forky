"use strict";
// This module loads the platform-specific build of the addon on
// the current system. The supported platforms are registered in
// the `platforms` object below, whose entries can be managed by
// by the Neon CLI:
//
//   https://www.npmjs.com/package/@neon-rs/cli
Object.defineProperty(exports, "__esModule", { value: true });
module.exports = require("@neon-rs/load").proxy({
    platforms: {
        "darwin-arm64": () => require("../platforms/darwin-arm64"),
        // "win32-x64-msvc": () => require("../platforms/win32-x64-msvc"),
        "linux-x64-gnu": () => require("../platforms/linux-x64-gnu"),
        "linux-x64-musl": () => require("../platforms/linux-x64-musl"),
    },
    debug: () => require("../index.node"),
});
