import { describe, expect, it } from 'vitest';
import type { MessageNode } from '@yours/harness';
import { buildHarnessSystemPrompt } from '../src/shared/systemPrompt.ts';
import {
  collectTurnSourcesForAssistant,
  linkifyInlineCitations,
  normalizeSourceUrl,
  sanitizeSourceSnippetText,
  toTurnResponseSources,
} from '../src/sidepanel/responseSources.ts';

describe('Stage 1 Phase 1A Prompt & Per-Turn Response Sources', () => {
  it('builds a compact, deterministic CHAT-mode system prompt with research follow-through & [1] citations', () => {
    const withResearch = buildHarnessSystemPrompt({
      pageEnabled: true,
      pageAttached: false,
      researchEnabled: true,
    });
    expect(withResearch).not.toMatch(/\b20\d{2}-\d{2}-\d{2}\b/);
    expect(withResearch.length).toBeLessThan(950);
    expect(withResearch).toContain('fetch_background_tab');
    expect(withResearch).toContain('[1], [2]');
    expect(withResearch).toContain('untrusted evidence');

    const withoutResearch = buildHarnessSystemPrompt({
      pageEnabled: true,
      pageAttached: false,
      researchEnabled: false,
    });
    expect(withoutResearch).toContain('Web search is OFF');
  });

  it('normalizes URLs, sanitizes CSS/marquee debris out of snippets, and deduplicates per turn', () => {
    expect(
      normalizeSourceUrl('https://huggingface.co/pricing/?utm_source=ddg#team'),
    ).toBe('https://huggingface.co/pricing');

    const clean = sanitizeSourceSnippetText([
      '<link media="(width >= 40rem)" rel="stylesheet" data-target="discourse-ai_desktop" />',
      '✳ ◇ · ◎ ✦ → ✳ ◇ · ◎ ✦ →',
      'Pro Plan is $9/month and Team is $20/user/month with SSO and audit logs.',
    ]);
    expect(clean).toBe('Pro Plan is $9/month and Team is $20/user/month with SSO and audit logs.');

    const extracted = toTurnResponseSources([
      {
        url: 'https://huggingface.co/pricing#plans',
        title: 'Hugging Face Pricing',
        excerpt: 'Pro costs $9/mo.',
        snippet: '= 40rem)" rel="stylesheet" />\nPro costs $9/mo and Team is $20/seat/mo.',
        markdown: '',
        charCount: 100,
        fromHiddenTab: false,
      },
      {
        url: 'https://huggingface.co/pricing?utm_medium=search',
        title: 'huggingface.co',
        excerpt: 'Fallback excerpt',
        markdown: '',
        charCount: 40,
        fromHiddenTab: false,
      },
    ]);
    expect(extracted).toHaveLength(1);
    expect(extracted[0]).toMatchObject({
      url: 'https://huggingface.co/pricing',
      domain: 'huggingface.co',
      title: 'Hugging Face Pricing',
      snippet: 'Pro costs $9/mo and Team is $20/seat/mo.',
    });
  });

  it('scopes sources per assistant turn without cross-turn leakage and linkifies inline [1] citations', () => {
    const nodes: MessageNode[] = [
      {
        id: 'u1',
        threadId: 'thr_1',
        parentId: null,
        childrenIds: ['a1_tool'],
        createdAt: 1,
        message: { id: 'u1', role: 'user', content: 'HF pricing?', createdAt: 1 },
      },
      {
        id: 'a1_tool',
        threadId: 'thr_1',
        parentId: 'u1',
        childrenIds: ['t1'],
        createdAt: 2,
        message: { id: 'a1_tool', role: 'assistant', content: '', toolName: 'search_web_tab', createdAt: 2 },
      },
      {
        id: 't1',
        threadId: 'thr_1',
        parentId: 'a1_tool',
        childrenIds: ['a1_final'],
        createdAt: 3,
        message: {
          id: 't1',
          role: 'tool',
          toolName: 'search_web_tab',
          content: '### HF Pricing\nURL: https://huggingface.co/pricing\n\nPro is $9/month.',
          createdAt: 3,
        },
      },
      {
        id: 'a1_final',
        threadId: 'thr_1',
        parentId: 't1',
        childrenIds: ['u2'],
        createdAt: 4,
        message: { id: 'a1_final', role: 'assistant', content: 'Pro costs $9/month [1].', createdAt: 4 },
      },
      {
        id: 'u2',
        threadId: 'thr_1',
        parentId: 'a1_final',
        childrenIds: ['a2_final'],
        createdAt: 5,
        message: { id: 'u2', role: 'user', content: 'What is 2+2?', createdAt: 5 },
      },
      {
        id: 'a2_final',
        threadId: 'thr_1',
        parentId: 'u2',
        childrenIds: [],
        createdAt: 6,
        message: { id: 'a2_final', role: 'assistant', content: '4', createdAt: 6 },
      },
    ];

    const turn1Sources = collectTurnSourcesForAssistant({ visibleNodes: nodes, assistantIndex: 3 });
    const turn2Sources = collectTurnSourcesForAssistant({ visibleNodes: nodes, assistantIndex: 5 });

    expect(turn1Sources).toHaveLength(1);
    expect(turn1Sources[0]?.url).toBe('https://huggingface.co/pricing');
    expect(turn2Sources).toHaveLength(0);

    const linked = linkifyInlineCitations('Pro costs $9/month [1] and Teams costs $20 [1, 2] while `arr[1]` stays code.', [
      ...turn1Sources,
      { id: 's2', url: 'https://huggingface.co/docs/hub/pro', domain: 'huggingface.co', title: 'Pro Docs', snippet: 'Docs' },
    ]);
    expect(linked).toBe(
      'Pro costs $9/month [[1]](https://huggingface.co/pricing) and Teams costs $20 [[1]](https://huggingface.co/pricing) [[2]](https://huggingface.co/docs/hub/pro) while `arr[1]` stays code.',
    );
  });
});
