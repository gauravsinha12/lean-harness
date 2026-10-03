import type { JsonSchema } from "./types.ts";

/**
 * A small JSON Schema validator covering what tool inputs use in practice: type, properties,
 * required, enum, items, additionalProperties, minimum/maximum. Errors are written for the model
 * to read and correct, e.g. `input.city: expected string, got number`.
 */
export function validate(value: unknown, schema: JsonSchema, path = "input"): string[] {
  const errors: string[] = [];
  const types = schema.type === undefined ? [] : Array.isArray(schema.type) ? schema.type : [schema.type];
  if (types.length && !types.some((t) => matchesType(value, t))) {
    errors.push(`${path}: expected ${types.join(" or ")}, got ${describe(value)}`);
    return errors;
  }
  if (schema.enum && !schema.enum.some((e) => e === value)) {
    errors.push(`${path}: must be one of ${schema.enum.map((e) => JSON.stringify(e)).join(", ")}`);
  }
  if (typeof value === "number") {
    if (schema.minimum !== undefined && value < schema.minimum) errors.push(`${path}: must be >= ${schema.minimum}`);
    if (schema.maximum !== undefined && value > schema.maximum) errors.push(`${path}: must be <= ${schema.maximum}`);
  }
  if (Array.isArray(value) && schema.items) {
    value.forEach((item, i) => errors.push(...validate(item, schema.items!, `${path}[${i}]`)));
  }
  if (isObject(value)) {
    for (const key of schema.required ?? []) {
      if (!(key in value)) errors.push(`${path}.${key}: required`);
    }
    for (const [key, sub] of Object.entries(schema.properties ?? {})) {
      if (key in value) errors.push(...validate(value[key], sub, `${path}.${key}`));
    }
    if (schema.additionalProperties === false) {
      for (const key of Object.keys(value)) {
        if (!(schema.properties ?? {})[key]) errors.push(`${path}.${key}: unknown property`);
      }
    }
  }
  return errors;
}

function matchesType(value: unknown, type: string): boolean {
  switch (type) {
    case "object": return isObject(value);
    case "array": return Array.isArray(value);
    case "integer": return Number.isInteger(value);
    case "number": return typeof value === "number" && Number.isFinite(value);
    case "null": return value === null;
    default: return typeof value === type;
  }
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function describe(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  return typeof v;
}
