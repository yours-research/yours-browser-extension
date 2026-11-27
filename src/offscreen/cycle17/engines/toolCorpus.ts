/**
 * Static drafting corpus shipped with Engine A: tool-call exemplars rendered
 * from the tool schemas in both wire dialects, plus chat-template scaffolding.
 * Generated purely from schemas (no benchmark responses), so it is fair to use
 * in every workload.
 */
import { EXTENSION_TOOLS, JsonStyle, LAB_TOOLS, SPACED_JSON, ToolParam, ToolSchema } from './toolGrammar.js';

function exampleValue(p: ToolParam): string {
  if (p.enum) return p.enum[0];
  switch (p.type) {
    case 'boolean':
      return 'false';
    case 'integer':
      return /max_?chars|maxChars/i.test(p.name) ? '24000' : /limit|max_results/i.test(p.name) ? '5' : '1';
    case 'number':
      return '1';
    case 'array':
      return '["https://example.com"]';
    default:
      return p.name === 'url' ? 'https://example.com' : '';
  }
}

export function renderXmlCall(t: ToolSchema, params: ToolParam[]): string {
  const body = params.map((p) => `<parameter=${p.name}>\n${exampleValue(p)}\n</parameter>\n`).join('');
  return `<tool_call>\n<function=${t.name}>\n${body}</function>\n</tool_call>`;
}

export function renderJsonCall(t: ToolSchema, params: ToolParam[], st: JsonStyle = SPACED_JSON): string {
  const args = params.map((p) => {
    const v = exampleValue(p);
    const lit = p.type === 'string' ? JSON.stringify(v) : v;
    return `"${p.name}"${st.colon}${lit}`;
  });
  return `<tool_call>{"name"${st.colon}"${t.name}"${st.comma}"arguments"${st.colon}{${args.join(st.comma)}}}</tool_call>`;
}

export function buildToolCorpus(tools: ToolSchema[] = [...LAB_TOOLS, ...EXTENSION_TOOLS], styles: JsonStyle[] = [SPACED_JSON]): string[] {
  const docs: string[] = [];
  for (const t of tools) {
    const req = t.params.filter((p) => p.required);
    const variants = [req, t.params.slice(0, Math.max(req.length, 1)), t.params];
    for (const v of variants) {
      docs.push(renderXmlCall(t, v) + '<|im_end|>');
      for (const st of styles) docs.push(renderJsonCall(t, v, st) + '<|im_end|>');
    }
  }
  docs.push('<|im_end|>\n<|im_start|>user\n<tool_response>\n');
  docs.push('\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n');
  docs.push('<|im_start|>assistant\n<think>\n\n</think>\n\n');
  docs.push('</think>\n\n');
  return docs;
}
