import type { JsCalcInput, JsCalcResult } from './types.ts';

type TokenType = 'number' | 'string' | 'ident' | 'op' | 'lparen' | 'rparen' | 'lbracket' | 'rbracket' | 'comma';

interface Token {
  type: TokenType;
  value: string;
}

type CalcValue = number | string | boolean | CalcValue[];

const LENGTH_TO_METERS: Record<string, number> = {
  mm: 0.001,
  millimeter: 0.001,
  millimeters: 0.001,
  cm: 0.01,
  centimeter: 0.01,
  centimeters: 0.01,
  m: 1,
  meter: 1,
  meters: 1,
  km: 1000,
  kilometer: 1000,
  kilometers: 1000,
  in: 0.0254,
  inch: 0.0254,
  inches: 0.0254,
  ft: 0.3048,
  foot: 0.3048,
  feet: 0.3048,
  yd: 0.9144,
  yard: 0.9144,
  yards: 0.9144,
  mi: 1609.344,
  mile: 1609.344,
  miles: 1609.344,
};

const MASS_TO_GRAMS: Record<string, number> = {
  mg: 0.001,
  milligram: 0.001,
  milligrams: 0.001,
  g: 1,
  gram: 1,
  grams: 1,
  kg: 1000,
  kilogram: 1000,
  kilograms: 1000,
  oz: 28.349523125,
  ounce: 28.349523125,
  ounces: 28.349523125,
  lb: 453.59237,
  lbs: 453.59237,
  pound: 453.59237,
  pounds: 453.59237,
};

const DATA_TO_BYTES: Record<string, number> = {
  B: 1,
  b: 1,
  KB: 1e3,
  kb: 1e3,
  KiB: 1024,
  kib: 1024,
  MB: 1e6,
  mb: 1e6,
  MiB: 1024 ** 2,
  mib: 1024 ** 2,
  GB: 1e9,
  gb: 1e9,
  GiB: 1024 ** 3,
  gib: 1024 ** 3,
  TB: 1e12,
  tb: 1e12,
  TiB: 1024 ** 4,
  tib: 1024 ** 4,
};

const TIME_TO_SECONDS: Record<string, number> = {
  ms: 0.001,
  millisecond: 0.001,
  milliseconds: 0.001,
  s: 1,
  sec: 1,
  secs: 1,
  second: 1,
  seconds: 1,
  min: 60,
  mins: 60,
  minute: 60,
  minutes: 60,
  hr: 3600,
  hrs: 3600,
  hour: 3600,
  hours: 3600,
  day: 86400,
  days: 86400,
};

function normalizeTempUnit(unit: string): 'C' | 'F' | 'K' | null {
  const lower = unit.trim().toLowerCase();
  if (lower === 'c' || lower === 'celsius' || lower === '°c') return 'C';
  if (lower === 'f' || lower === 'fahrenheit' || lower === '°f') return 'F';
  if (lower === 'k' || lower === 'kelvin') return 'K';
  return null;
}

export function convertUnit(val: number, fromUnit: string, toUnit: string): number {
  if (!Number.isFinite(val)) {
    throw new Error('Value to convert must be a finite number.');
  }
  const fromTrim = fromUnit.trim();
  const toTrim = toUnit.trim();
  const fromLower = fromTrim.toLowerCase();
  const toLower = toTrim.toLowerCase();

  // Temperature
  const fromTemp = normalizeTempUnit(fromTrim);
  const toTemp = normalizeTempUnit(toTrim);
  if (fromTemp && toTemp) {
    let celsius = val;
    if (fromTemp === 'F') celsius = ((val - 32) * 5) / 9;
    else if (fromTemp === 'K') celsius = val - 273.15;

    if (toTemp === 'C') return celsius;
    if (toTemp === 'F') return (celsius * 9) / 5 + 32;
    return celsius + 273.15;
  }

  // Data (preserve exact case for KiB/KB first, fallback to lowercase)
  const dataFrom = DATA_TO_BYTES[fromTrim] ?? DATA_TO_BYTES[fromLower];
  const dataTo = DATA_TO_BYTES[toTrim] ?? DATA_TO_BYTES[toLower];
  if (dataFrom !== undefined && dataTo !== undefined) {
    return (val * dataFrom) / dataTo;
  }

  const tables: Array<Record<string, number>> = [LENGTH_TO_METERS, MASS_TO_GRAMS, TIME_TO_SECONDS];
  for (const table of tables) {
    if (table[fromLower] !== undefined && table[toLower] !== undefined) {
      return (val * table[fromLower]) / table[toLower];
    }
  }

  throw new Error(`Unsupported or incompatible unit conversion: "${fromUnit}" -> "${toUnit}".`);
}

function parseIsoDateUtcMs(iso: string): number {
  const trimmed = iso.trim();
  if (!trimmed) {
    throw new Error('Date string cannot be empty.');
  }
  const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(trimmed);
  if (dateOnly) {
    const year = Number.parseInt(dateOnly[1], 10);
    const month = Number.parseInt(dateOnly[2], 10) - 1;
    const day = Number.parseInt(dateOnly[3], 10);
    const ms = Date.UTC(year, month, day);
    const check = new Date(ms);
    if (check.getUTCFullYear() !== year || check.getUTCMonth() !== month || check.getUTCDate() !== day) {
      throw new Error(`Invalid calendar date: "${iso}".`);
    }
    return ms;
  }
  const parsed = Date.parse(trimmed);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid ISO date: "${iso}".`);
  }
  return parsed;
}

function calcDaysBetween(isoA: string, isoB: string): number {
  const msA = parseIsoDateUtcMs(isoA);
  const msB = parseIsoDateUtcMs(isoB);
  const diffDays = (msB - msA) / 86_400_000;
  return Math.round(diffDays * 1e9) / 1e9;
}

function calcAddDays(iso: string, days: number): string {
  if (!Number.isFinite(days)) {
    throw new Error('Days offset must be a finite number.');
  }
  const baseMs = parseIsoDateUtcMs(iso);
  const target = new Date(baseMs + days * 86_400_000);
  if (/^\d{4}-\d{2}-\d{2}$/.test(iso.trim()) && Number.isInteger(days)) {
    return target.toISOString().slice(0, 10);
  }
  return target.toISOString();
}

function tokenizeExpr(expr: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < expr.length) {
    const ch = expr[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }

    // Numbers including scientific notation (e.g. 1.5e-3, .5, 42)
    if (/[0-9]/.test(ch) || (ch === '.' && i + 1 < expr.length && /[0-9]/.test(expr[i + 1]))) {
      const rest = expr.slice(i);
      const numMatch = /^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/.exec(rest);
      if (!numMatch) {
        throw new Error(`Invalid number near "${rest.slice(0, 10)}".`);
      }
      tokens.push({ type: 'number', value: numMatch[0] });
      i += numMatch[0].length;
      continue;
    }

    // String literals
    if (ch === '"' || ch === "'") {
      const quote = ch;
      let str = '';
      i++;
      let closed = false;
      while (i < expr.length) {
        const c = expr[i];
        if (c === '\\' && i + 1 < expr.length) {
          str += expr[i + 1];
          i += 2;
          continue;
        }
        if (c === quote) {
          closed = true;
          i++;
          break;
        }
        str += c;
        i++;
      }
      if (!closed) {
        throw new Error('Unterminated string literal in expression.');
      }
      tokens.push({ type: 'string', value: str });
      continue;
    }

    // Identifiers
    if (/[a-zA-Z_]/.test(ch)) {
      const idMatch = /^[a-zA-Z_][a-zA-Z0-9_]*/.exec(expr.slice(i));
      if (idMatch) {
        tokens.push({ type: 'ident', value: idMatch[0] });
        i += idMatch[0].length;
        continue;
      }
    }

    if (ch === '+' || ch === '-' || ch === '*' || ch === '/' || ch === '%' || ch === '^') {
      tokens.push({ type: 'op', value: ch });
      i++;
      continue;
    }
    if (ch === '(') {
      tokens.push({ type: 'lparen', value: ch });
      i++;
      continue;
    }
    if (ch === ')') {
      tokens.push({ type: 'rparen', value: ch });
      i++;
      continue;
    }
    if (ch === '[') {
      tokens.push({ type: 'lbracket', value: ch });
      i++;
      continue;
    }
    if (ch === ']') {
      tokens.push({ type: 'rbracket', value: ch });
      i++;
      continue;
    }
    if (ch === ',') {
      tokens.push({ type: 'comma', value: ch });
      i++;
      continue;
    }

    throw new Error(`Unexpected character "${ch}" at position ${i}.`);
  }

  return tokens;
}

function asNumber(val: CalcValue, context: string): number {
  if (typeof val !== 'number' || !Number.isFinite(val)) {
    throw new Error(`Expected finite number for ${context}.`);
  }
  return val;
}

function asString(val: CalcValue, context: string): string {
  if (typeof val !== 'string') {
    throw new Error(`Expected string for ${context}.`);
  }
  return val;
}

function flattenNumbers(args: CalcValue[], fnName: string): number[] {
  const nums: number[] = [];
  for (const arg of args) {
    if (Array.isArray(arg)) {
      for (const item of arg) {
        nums.push(asNumber(item, fnName));
      }
    } else {
      nums.push(asNumber(arg, fnName));
    }
  }
  return nums;
}

function invokeFunction(name: string, args: CalcValue[]): CalcValue {
  const fn = name.toLowerCase();

  switch (fn) {
    case 'sqrt':
      if (args.length !== 1) throw new Error('sqrt(x) requires 1 argument.');
      return Math.sqrt(asNumber(args[0], 'sqrt'));
    case 'abs':
      if (args.length !== 1) throw new Error('abs(x) requires 1 argument.');
      return Math.abs(asNumber(args[0], 'abs'));
    case 'round': {
      if (args.length < 1 || args.length > 2) throw new Error('round(x, digits?) takes 1 or 2 arguments.');
      const x = asNumber(args[0], 'round');
      if (args.length === 2) {
        const digits = Math.trunc(asNumber(args[1], 'round digits'));
        const factor = 10 ** digits;
        return Math.round(x * factor) / factor;
      }
      return Math.round(x);
    }
    case 'floor':
      if (args.length !== 1) throw new Error('floor(x) requires 1 argument.');
      return Math.floor(asNumber(args[0], 'floor'));
    case 'ceil':
      if (args.length !== 1) throw new Error('ceil(x) requires 1 argument.');
      return Math.ceil(asNumber(args[0], 'ceil'));
    case 'sin':
      if (args.length !== 1) throw new Error('sin(x) requires 1 argument.');
      return Math.sin(asNumber(args[0], 'sin'));
    case 'cos':
      if (args.length !== 1) throw new Error('cos(x) requires 1 argument.');
      return Math.cos(asNumber(args[0], 'cos'));
    case 'tan':
      if (args.length !== 1) throw new Error('tan(x) requires 1 argument.');
      return Math.tan(asNumber(args[0], 'tan'));
    case 'log':
      if (args.length !== 1) throw new Error('log(x) requires 1 argument.');
      return Math.log(asNumber(args[0], 'log'));
    case 'log10':
      if (args.length !== 1) throw new Error('log10(x) requires 1 argument.');
      return Math.log10(asNumber(args[0], 'log10'));
    case 'exp':
      if (args.length !== 1) throw new Error('exp(x) requires 1 argument.');
      return Math.exp(asNumber(args[0], 'exp'));
    case 'pow':
      if (args.length !== 2) throw new Error('pow(base, exp) requires 2 arguments.');
      return Math.pow(asNumber(args[0], 'pow base'), asNumber(args[1], 'pow exp'));
    case 'min': {
      const nums = flattenNumbers(args, 'min');
      if (nums.length === 0) throw new Error('min() requires at least 1 number.');
      return Math.min(...nums);
    }
    case 'max': {
      const nums = flattenNumbers(args, 'max');
      if (nums.length === 0) throw new Error('max() requires at least 1 number.');
      return Math.max(...nums);
    }
    case 'sum': {
      const nums = flattenNumbers(args, 'sum');
      return nums.reduce((acc, n) => acc + n, 0);
    }
    case 'mean':
    case 'avg': {
      const nums = flattenNumbers(args, fn);
      if (nums.length === 0) throw new Error(`${fn}() requires at least 1 number.`);
      return nums.reduce((acc, n) => acc + n, 0) / nums.length;
    }
    case 'median': {
      const nums = flattenNumbers(args, 'median').sort((a, b) => a - b);
      if (nums.length === 0) throw new Error('median() requires at least 1 number.');
      const mid = Math.floor(nums.length / 2);
      return nums.length % 2 === 1 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
    }
    case 'std': {
      const nums = flattenNumbers(args, 'std');
      if (nums.length === 0) throw new Error('std() requires at least 1 number.');
      const meanVal = nums.reduce((acc, n) => acc + n, 0) / nums.length;
      const variance = nums.reduce((acc, n) => acc + (n - meanVal) ** 2, 0) / nums.length;
      return Math.sqrt(variance);
    }
    case 'count': {
      if (args.length === 1 && Array.isArray(args[0])) {
        return args[0].length;
      }
      return args.length;
    }
    case 'convert':
      if (args.length !== 3) throw new Error('convert(val, from, to) requires 3 arguments.');
      return convertUnit(
        asNumber(args[0], 'convert value'),
        asString(args[1], 'convert from'),
        asString(args[2], 'convert to'),
      );
    case 'daysbetween':
      if (args.length !== 2) throw new Error('daysBetween(isoA, isoB) requires 2 arguments.');
      return calcDaysBetween(asString(args[0], 'dateA'), asString(args[1], 'dateB'));
    case 'adddays':
      if (args.length !== 2) throw new Error('addDays(iso, n) requires 2 arguments.');
      return calcAddDays(asString(args[0], 'iso'), asNumber(args[1], 'days'));
    default:
      throw new Error(`Unknown function "${name}".`);
  }
}

function evaluateExpression(expr: string): CalcValue {
  const tokens = tokenizeExpr(expr);
  if (tokens.length === 0) {
    throw new Error('Expression is empty.');
  }

  let cursor = 0;
  const peek = () => tokens[cursor];
  const consume = () => tokens[cursor++];

  function parseAddSub(): CalcValue {
    let left = parseMulDiv();
    while (peek()?.type === 'op' && (peek().value === '+' || peek().value === '-')) {
      const op = consume().value;
      const right = parseMulDiv();
      const l = asNumber(left, op);
      const r = asNumber(right, op);
      left = op === '+' ? l + r : l - r;
    }
    return left;
  }

  function parseMulDiv(): CalcValue {
    let left = parseUnary();
    while (peek()?.type === 'op' && (peek().value === '*' || peek().value === '/' || peek().value === '%')) {
      const op = consume().value;
      const right = parseUnary();
      const l = asNumber(left, op);
      const r = asNumber(right, op);
      if ((op === '/' || op === '%') && r === 0) {
        throw new Error('Division by zero.');
      }
      left = op === '*' ? l * r : op === '/' ? l / r : l % r;
    }
    return left;
  }

  function parseUnary(): CalcValue {
    if (peek()?.type === 'op' && (peek().value === '+' || peek().value === '-')) {
      const op = consume().value;
      const val = asNumber(parseUnary(), `unary ${op}`);
      return op === '-' ? -val : val;
    }
    return parsePower();
  }

  function parsePower(): CalcValue {
    const base = parsePrimary();
    if (peek()?.type === 'op' && peek().value === '^') {
      consume();
      const exponent = parseUnary();
      return Math.pow(asNumber(base, '^ base'), asNumber(exponent, '^ exponent'));
    }
    return base;
  }

  function parsePrimary(): CalcValue {
    const tok = peek();
    if (!tok) {
      throw new Error('Unexpected end of expression.');
    }

    if (tok.type === 'number') {
      consume();
      const num = Number(tok.value);
      if (!Number.isFinite(num)) {
        throw new Error(`Invalid numeric literal "${tok.value}".`);
      }
      return num;
    }

    if (tok.type === 'string') {
      consume();
      return tok.value;
    }

    if (tok.type === 'lparen') {
      consume();
      const val = parseAddSub();
      if (peek()?.type !== 'rparen') {
        throw new Error('Expected closing ")".');
      }
      consume();
      return val;
    }

    if (tok.type === 'lbracket') {
      consume();
      const items: CalcValue[] = [];
      if (peek()?.type !== 'rbracket') {
        items.push(parseAddSub());
        while (peek()?.type === 'comma') {
          consume();
          items.push(parseAddSub());
        }
      }
      if (peek()?.type !== 'rbracket') {
        throw new Error('Expected closing "]".');
      }
      consume();
      return items;
    }

    if (tok.type === 'ident') {
      consume();
      if (peek()?.type === 'lparen') {
        consume();
        const args: CalcValue[] = [];
        if (peek()?.type !== 'rparen') {
          args.push(parseAddSub());
          while (peek()?.type === 'comma') {
            consume();
            args.push(parseAddSub());
          }
        }
        if (peek()?.type !== 'rparen') {
          throw new Error(`Expected closing ")" after arguments to ${tok.value}().`);
        }
        consume();
        return invokeFunction(tok.value, args);
      }

      const lower = tok.value.toLowerCase();
      if (lower === 'pi') return Math.PI;
      if (lower === 'e') return Math.E;
      if (lower === 'true') return true;
      if (lower === 'false') return false;
      throw new Error(`Unknown identifier "${tok.value}".`);
    }

    throw new Error(`Unexpected token "${tok.value}".`);
  }

  const value = parseAddSub();
  if (cursor < tokens.length) {
    throw new Error(`Unexpected trailing token "${tokens[cursor].value}".`);
  }
  return value;
}

function queryJsonPath(rawData: unknown, path?: string): unknown {
  let obj: unknown = rawData;
  if (typeof rawData === 'string') {
    obj = JSON.parse(rawData);
  }
  if (!path || !path.trim()) {
    return obj;
  }

  const cleanedPath = path.trim().replace(/^\$\.?/, '');
  if (!cleanedPath) {
    return obj;
  }

  const segments: Array<string | number> = [];
  const segRegex = /([^.[\]]+)|\[(\d+|"[^"]*"|'[^']*')\]/g;
  let match: RegExpExecArray | null;
  while ((match = segRegex.exec(cleanedPath)) !== null) {
    if (match[1] !== undefined) {
      segments.push(match[1]);
    } else if (match[2] !== undefined) {
      const idxRaw = match[2];
      if (/^\d+$/.test(idxRaw)) {
        segments.push(Number.parseInt(idxRaw, 10));
      } else {
        segments.push(idxRaw.slice(1, -1));
      }
    }
  }

  let current: unknown = obj;
  for (const seg of segments) {
    if (current === null || current === undefined) {
      return undefined;
    }
    if (typeof seg === 'number' && Array.isArray(current)) {
      current = current[seg];
    } else if (typeof current === 'object') {
      current = (current as Record<string, unknown>)[String(seg)];
    } else {
      return undefined;
    }
  }
  return current;
}

function parseCsvRows(csvText: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;

  for (let i = 0; i < csvText.length; i++) {
    const ch = csvText[i];
    if (inQuotes) {
      if (ch === '"' && csvText[i + 1] === '"') {
        field += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      row.push(field.trim());
      field = '';
    } else if (ch === '\r') {
      if (csvText[i + 1] === '\n') {
        i++;
      }
      row.push(field.trim());
      if (row.some((c) => c !== '')) {
        rows.push(row);
      }
      row = [];
      field = '';
    } else if (ch === '\n') {
      row.push(field.trim());
      if (row.some((c) => c !== '')) {
        rows.push(row);
      }
      row = [];
      field = '';
    } else {
      field += ch;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field.trim());
    if (row.some((c) => c !== '')) {
      rows.push(row);
    }
  }

  return rows;
}

interface CsvColumnStats {
  count: number;
  min: number;
  max: number;
  mean: number;
}

function summarizeCsv(csvText: string): {
  rowCount: number;
  columns: string[];
  numericStats: Record<string, CsvColumnStats>;
} {
  const rows = parseCsvRows(csvText);
  if (rows.length === 0) {
    throw new Error('CSV input is empty.');
  }

  const columns = rows[0];
  const dataRows = rows.slice(1);
  const numericStats: Record<string, CsvColumnStats> = {};

  for (let colIdx = 0; colIdx < columns.length; colIdx++) {
    const colName = columns[colIdx] || `col_${colIdx}`;
    const values: number[] = [];
    for (const r of dataRows) {
      const cell = r[colIdx];
      if (cell !== undefined && cell !== '') {
        const num = Number(cell);
        if (Number.isFinite(num)) {
          values.push(num);
        }
      }
    }
    if (values.length > 0) {
      const sum = values.reduce((a, b) => a + b, 0);
      numericStats[colName] = {
        count: values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        mean: Math.round((sum / values.length) * 1e6) / 1e6,
      };
    }
  }

  return {
    rowCount: dataRows.length,
    columns,
    numericStats,
  };
}

function formatCalcValue(val: unknown): string {
  if (typeof val === 'number') {
    if (Number.isInteger(val)) {
      return String(val);
    }
    // Avoid floating-point noise like 0.30000000000000004 while preserving precision
    const rounded = Number(val.toPrecision(12));
    return String(rounded);
  }
  if (typeof val === 'string') {
    return val;
  }
  return JSON.stringify(val);
}

export function runJsCalc(input: JsCalcInput): JsCalcResult {
  try {
    const mode = input.mode ?? 'expr';

    if (mode === 'expr') {
      if (typeof input.expression !== 'string' || !input.expression.trim()) {
        return { ok: false, formatted: '', error: 'Missing expression for mode "expr".' };
      }
      const value = evaluateExpression(input.expression);
      return {
        ok: true,
        result: value,
        formatted: formatCalcValue(value),
      };
    }

    if (mode === 'unit') {
      const numVal = typeof input.value === 'number' ? input.value : Number(input.value);
      if (!Number.isFinite(numVal) || !input.from || !input.to) {
        return { ok: false, formatted: '', error: 'Unit mode requires numeric value, from, and to.' };
      }
      const converted = convertUnit(numVal, input.from, input.to);
      return {
        ok: true,
        result: converted,
        formatted: `${formatCalcValue(converted)} ${input.to}`,
      };
    }

    if (mode === 'date') {
      if (input.dateA && input.dateB) {
        const diff = calcDaysBetween(input.dateA, input.dateB);
        return {
          ok: true,
          result: diff,
          formatted: `${formatCalcValue(diff)} days`,
        };
      }
      const baseDate = input.dateA ?? (typeof input.value === 'string' ? input.value : undefined);
      if (baseDate && typeof input.days === 'number') {
        const nextDate = calcAddDays(baseDate, input.days);
        return {
          ok: true,
          result: nextDate,
          formatted: nextDate,
        };
      }
      return {
        ok: false,
        formatted: '',
        error: 'Date mode requires (dateA, dateB) or (dateA, days).',
      };
    }

    if (mode === 'json') {
      const source = input.data ?? input.value;
      if (source === undefined) {
        return { ok: false, formatted: '', error: 'JSON mode requires data or value.' };
      }
      const extracted = queryJsonPath(source, input.path);
      return {
        ok: true,
        result: extracted,
        formatted: extracted === undefined ? 'undefined' : formatCalcValue(extracted),
      };
    }

    if (mode === 'csv') {
      if (typeof input.data !== 'string' || !input.data.trim()) {
        return { ok: false, formatted: '', error: 'CSV mode requires non-empty data string.' };
      }
      const summary = summarizeCsv(input.data);
      const statLines = Object.entries(summary.numericStats).map(
        ([col, s]) => `${col}: min=${s.min}, max=${s.max}, mean=${s.mean} (n=${s.count})`,
      );
      const formatted = [
        `Rows: ${summary.rowCount} | Columns (${summary.columns.length}): ${summary.columns.join(', ')}`,
        ...statLines,
      ].join('\n');
      return {
        ok: true,
        result: summary,
        formatted,
      };
    }

    return { ok: false, formatted: '', error: `Unsupported js_calc mode: "${String(mode)}".` };
  } catch (err: unknown) {
    return {
      ok: false,
      formatted: '',
      error: err instanceof Error ? err.message : String(err),
    };
  }
}
