export interface ThreadTextMatch {
  range: Range;
  element: HTMLElement;
}

/**
 * Walks visible text nodes inside `container` (excluding `.yui-find-bar` controls)
 * and collects ordered DOM `Range` objects + enclosing elements for `query`.
 */
export function collectFindMatches(
  container: HTMLElement | null,
  query: string,
  caseSensitive: boolean,
): ThreadTextMatch[] {
  const needleRaw = query;
  if (!container || needleRaw.length === 0) {
    return [];
  }

  const needle = caseSensitive ? needleRaw : needleRaw.toLowerCase();
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      if (parent.closest('.yui-find-bar, button, svg, [aria-hidden="true"]')) {
        return NodeFilter.FILTER_REJECT;
      }
      if (!node.nodeValue || node.nodeValue.length === 0) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  const matches: ThreadTextMatch[] = [];
  let current = walker.nextNode();
  while (current) {
    const textNode = current as Text;
    const raw = textNode.nodeValue ?? '';
    const haystack = caseSensitive ? raw : raw.toLowerCase();
    let offset = 0;
    while (offset <= haystack.length - needle.length) {
      const idx = haystack.indexOf(needle, offset);
      if (idx < 0) break;
      const range = document.createRange();
      range.setStart(textNode, idx);
      range.setEnd(textNode, idx + needle.length);
      const element =
        textNode.parentElement?.closest<HTMLElement>(
          '.yui-assistant-turn, .yui-user-bubble, .yui-tool-card, .ext-tool-group',
        ) ??
        textNode.parentElement ??
        container;
      matches.push({ range, element });
      offset = idx + Math.max(1, needle.length);
    }
    current = walker.nextNode();
  }
  return matches;
}

/**
 * Counts substring occurrences across an arbitrary string list (pure fallback for unit tests).
 */
export function countQueryOccurrences(
  texts: readonly string[],
  query: string,
  caseSensitive: boolean,
): number {
  if (!query) return 0;
  const needle = caseSensitive ? query : query.toLowerCase();
  let total = 0;
  for (const text of texts) {
    const hay = caseSensitive ? text : text.toLowerCase();
    let pos = 0;
    while (pos <= hay.length - needle.length) {
      const idx = hay.indexOf(needle, pos);
      if (idx < 0) break;
      total += 1;
      pos = idx + Math.max(1, needle.length);
    }
  }
  return total;
}
