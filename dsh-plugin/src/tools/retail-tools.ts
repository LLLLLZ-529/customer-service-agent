/**
 * Retail Tools: 16 tau-bench retail customer service tools.
 * Each tool calls the Python tool bridge over HTTP.
 */
import { defineTool } from "@deepseek-ai/dsh-tools";

const BRIDGE_URL = "http://127.0.0.1:9000/execute";

/** Call the Python tool bridge. */
async function callBridge(toolName: string, args: Record<string, unknown>) {
  const resp = await fetch(BRIDGE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool_name: toolName, arguments: args }),
  });
  const data = await resp.json();
  if (data.error) throw new Error(data.error);
  return JSON.stringify(data.result);
}

// ── User lookup ──────────────────────────────────────────────

export const findUserByEmail = defineTool({
  name: "find_user_id_by_email",
  description: "Find user id by email. If not found, returns an error. Try this first before other lookup methods.",
  parameters: {
    email: { type: "string", required: true, description: "The email address of the user." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("find_user_id_by_email", args);
  },
});

export const findUserByNameZip = defineTool({
  name: "find_user_id_by_name_zip",
  description: "Find user id by first name, last name, and zip code. Only call if email lookup fails or user cannot remember email.",
  parameters: {
    first_name: { type: "string", required: true, description: "First name." },
    last_name: { type: "string", required: true, description: "Last name." },
    zip: { type: "string", required: true, description: "Zip code." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("find_user_id_by_name_zip", args);
  },
});

// ── Data retrieval ───────────────────────────────────────────

export const getOrderDetails = defineTool({
  name: "get_order_details",
  description: "Get the status and details of an order.",
  parameters: {
    order_id: { type: "string", required: true, description: "Order id like '#W0000000', note the '#' prefix." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("get_order_details", args);
  },
});

export const getProductDetails = defineTool({
  name: "get_product_details",
  description: "Get product details including available variants (item ids, options, prices).",
  parameters: {
    product_id: { type: "string", required: true, description: "Product id, e.g. '1656367028'." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("get_product_details", args);
  },
});

export const getUserDetails = defineTool({
  name: "get_user_details",
  description: "Get user details including address, payment methods, and order history.",
  parameters: {
    user_id: { type: "string", required: true, description: "User id." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("get_user_details", args);
  },
});

export const listAllProductTypes = defineTool({
  name: "list_all_product_types",
  description: "List all available product types with names and product ids.",
  parameters: {},
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("list_all_product_types", args);
  },
});

// ── Order operations ─────────────────────────────────────────

export const cancelPendingOrder = defineTool({
  name: "cancel_pending_order",
  description: "Cancel a pending order. Only works for orders in 'pending' status.",
  parameters: {
    order_id: { type: "string", required: true, description: "Order id like '#W0000000'." },
    reason: { type: "string", required: false, description: "Cancellation reason." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("cancel_pending_order", args);
  },
});

export const modifyPendingOrderItems = defineTool({
  name: "modify_pending_order_items",
  description: "Modify items in a pending order (add, remove, or change quantities).",
  parameters: {
    order_id: { type: "string", required: true, description: "Order id." },
    items: { type: "array", required: true, description: "New list of items with item_id and quantity." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("modify_pending_order_items", args);
  },
});

export const modifyPendingOrderAddress = defineTool({
  name: "modify_pending_order_address",
  description: "Modify the shipping address of a pending order.",
  parameters: {
    order_id: { type: "string", required: true, description: "Order id." },
    address: { type: "object", required: true, description: "New address object." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("modify_pending_order_address", args);
  },
});

export const modifyPendingOrderPayment = defineTool({
  name: "modify_pending_order_payment",
  description: "Modify the payment method for a pending order.",
  parameters: {
    order_id: { type: "string", required: true, description: "Order id." },
    payment_method_id: { type: "string", required: true, description: "New payment method id." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("modify_pending_order_payment", args);
  },
});

// ── Return / Exchange ────────────────────────────────────────

export const returnDeliveredOrderItems = defineTool({
  name: "return_delivered_order_items",
  description: "Process a return for items in a delivered order.",
  parameters: {
    order_id: { type: "string", required: true, description: "Order id." },
    items: { type: "array", required: true, description: "Items to return: item_id and quantity." },
    refund_method: { type: "string", required: true, description: "Refund method: 'original' or other." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("return_delivered_order_items", args);
  },
});

export const exchangeDeliveredOrderItems = defineTool({
  name: "exchange_delivered_order_items",
  description: "Process an exchange for items in a delivered order. May involve price difference refund/charge.",
  parameters: {
    order_id: { type: "string", required: true, description: "Order id." },
    items: { type: "array", required: true, description: "Exchange items: old item_id, new item_id, quantity." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("exchange_delivered_order_items", args);
  },
});

// ── User account ─────────────────────────────────────────────

export const modifyUserAddress = defineTool({
  name: "modify_user_address",
  description: "Modify the default address on the user's account.",
  parameters: {
    user_id: { type: "string", required: true, description: "User id." },
    address: { type: "object", required: true, description: "New address." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("modify_user_address", args);
  },
});

// ── Utility ──────────────────────────────────────────────────

export const calculate = defineTool({
  name: "calculate",
  description: "Perform a mathematical calculation. Use for price differences, totals, etc.",
  parameters: {
    expression: { type: "string", required: true, description: "Math expression, e.g. '272.33 - 255.70'." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("calculate", args);
  },
});

export const think = defineTool({
  name: "think",
  description: "Record reasoning/notes. Does not affect the task. Use to plan steps.",
  parameters: {
    thought: { type: "string", required: true, description: "Your reasoning." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute() {
    return "thought recorded";
  },
});

export const transferToHuman = defineTool({
  name: "transfer_to_human_agents",
  description: "Transfer the conversation to a human agent. Use only when the request is outside allowed operations.",
  parameters: {
    summary: { type: "string", required: true, description: "Summary of the issue for the human agent." },
  },
  output: {
    schema: { type: "string" },
    render: (_a, v) => [{ type: "text", text: v }],
  },
  async execute(args) {
    return callBridge("transfer_to_human_agents", args);
  },
});

/** All retail tools, for registration. */
export const ALL_RETAIL_TOOLS = [
  findUserByEmail,
  findUserByNameZip,
  getOrderDetails,
  getProductDetails,
  getUserDetails,
  listAllProductTypes,
  cancelPendingOrder,
  modifyPendingOrderItems,
  modifyPendingOrderAddress,
  modifyPendingOrderPayment,
  returnDeliveredOrderItems,
  exchangeDeliveredOrderItems,
  modifyUserAddress,
  calculate,
  think,
  transferToHuman,
];
