/**
 * DSH Plugin: Retail Customer Service Tools
 * Registers 16 tau-bench retail tools into DeepSeek Harness.
 *
 * Tools are backed by a Python tool bridge (tool_bridge.py) which executes
 * them in the tau-bench retail simulation environment.
 */
import type { Context } from "@deepseek-ai/cordis";
import { ALL_RETAIL_TOOLS } from "./tools/retail-tools.js";

export const name = "retail-tools";
export const inject = ["tools"];

export interface Config {
  bridgeUrl: string;
}

export const Config = {
  bridgeUrl: {
    type: "string",
    description: "URL of the Python tool bridge server.",
    default: "http://127.0.0.1:9000",
  },
};

export function apply(ctx: Context<Config>) {
  const count = ALL_RETAIL_TOOLS.length;
  ctx.logger.info(`[retail-tools] registering ${count} retail tools`);

  for (const tool of ALL_RETAIL_TOOLS) {
    ctx.tools.register(tool);
  }

  ctx.logger.info(`[retail-tools] ${count} tools registered successfully`);

  // Cleanup on unmount
  ctx.on("dispose", () => {
    ctx.logger.info("[retail-tools] plugin unloaded");
  });
}
