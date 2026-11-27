import { useState } from 'react';
import {
  CodeBlockCard,
  StreamingResponse,
  type StreamingResponseStatus,
} from '@yours/ui';
import {
  ChecklistCard,
  ChoicePromptCard,
  CompareTable,
  DraftCard,
  EntitySummaryCard,
  MetricStripCard,
  PageActionCard,
  PageBriefCard,
  QuoteExcerptCard,
  TimelineCard,
  type ChecklistItem,
  type ChoiceOption,
  type CompareTableRow,
  type DraftToneOption,
  type EntityField,
  type PageActionField,
  type TimelineStep,
} from '@yours/ui/genui';

import { linkifyInlineCitations, type TurnResponseSource } from '../responseSources.ts';

export interface GenUiRendererProps {
  content: string;
  status?: StreamingResponseStatus;
  streamingSpeed?: number;
  sources?: readonly TurnResponseSource[];
  onCopyText?: (text: string) => void;
  onSaveText?: (text: string) => void;
  onSelectChoice?: (question: string, choiceId: string) => void;
  onConfirmAction?: (title: string) => void;
}

type ParsedSegment =
  | { kind: 'text'; value: string }
  | { kind: 'code'; language: string; code: string }
  | { kind: 'genui'; payload: Record<string, unknown>; raw: string };

type RenderSegment =
  | { kind: 'markdown'; value: string }
  | { kind: 'code'; language: string; value: string }
  | { kind: 'genui'; payload: Record<string, unknown>; raw: string };

const FENCED_BLOCK_RE = /```([a-zA-Z0-9_-]*)\r?\n([\s\S]*?)```/g;

export function parseAssistantSegments(content: string): ParsedSegment[] {
  const segments: ParsedSegment[] = [];
  let cursor = 0;
  FENCED_BLOCK_RE.lastIndex = 0;

  let match: RegExpExecArray | null;
  while ((match = FENCED_BLOCK_RE.exec(content)) !== null) {
    if (match.index > cursor) {
      segments.push({ kind: 'text', value: content.slice(cursor, match.index) });
    }
    const lang = (match[1] ?? '').trim().toLowerCase();
    const body = match[2] ?? '';

    if (lang === 'genui' || lang === 'yours-genui') {
      try {
        const parsed = JSON.parse(body) as unknown;
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          segments.push({
            kind: 'genui',
            payload: parsed as Record<string, unknown>,
            raw: body,
          });
        } else {
          segments.push({ kind: 'code', language: lang || 'json', code: body.trimEnd() });
        }
      } catch {
        segments.push({ kind: 'code', language: lang || 'json', code: body.trimEnd() });
      }
    } else {
      segments.push({ kind: 'code', language: lang || 'text', code: body.trimEnd() });
    }

    cursor = match.index + match[0].length;
  }

  if (cursor < content.length) {
    segments.push({ kind: 'text', value: content.slice(cursor) });
  }

  return segments;
}

function splitForStreamingRenderer(
  content: string,
  status?: StreamingResponseStatus,
): RenderSegment[] {
  const result: RenderSegment[] = [];
  let mdStart = 0;
  FENCED_BLOCK_RE.lastIndex = 0;

  let match: RegExpExecArray | null;
  while ((match = FENCED_BLOCK_RE.exec(content)) !== null) {
    const lang = (match[1] ?? '').trim().toLowerCase();
    const body = match[2] ?? '';
    if (lang === 'genui' || lang === 'yours-genui') {
      let parsed: unknown = null;
      try {
        parsed = JSON.parse(body);
      } catch {
        continue;
      }
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        continue;
      }
      const before = content.slice(mdStart, match.index);
      if (before.trim().length > 0) {
        result.push({ kind: 'markdown', value: before });
      }
      result.push({
        kind: 'genui',
        payload: parsed as Record<string, unknown>,
        raw: body,
      });
      mdStart = match.index + match[0].length;
    } else if (status !== 'streaming') {
      const before = content.slice(mdStart, match.index);
      if (before.trim().length > 0) {
        result.push({ kind: 'markdown', value: before });
      }
      result.push({
        kind: 'code',
        language: lang || 'text',
        value: body.trimEnd(),
      });
      mdStart = match.index + match[0].length;
    }
  }

  const tail = content.slice(mdStart);
  if (tail.trim().length > 0 || (result.length === 0 && tail.length > 0)) {
    result.push({ kind: 'markdown', value: tail });
  }
  return result;
}

function InteractiveChecklist({
  title,
  initialItems,
  onCopy,
}: {
  title: string;
  initialItems: ChecklistItem[];
  onCopy?: () => void;
}) {
  const [items, setItems] = useState<ChecklistItem[]>(initialItems);
  return (
    <ChecklistCard
      title={title}
      items={items}
      onToggleItem={(id) => {
        setItems((prev) => prev.map((it) => (it.id === id ? { ...it, done: !it.done } : it)));
      }}
      onCopy={onCopy}
    />
  );
}

function InteractiveDraft({
  title,
  tones,
  onCopy,
  onPrimaryAction,
  primaryActionLabel,
}: {
  title?: string;
  tones: DraftToneOption[];
  onCopy?: (text: string) => void;
  onPrimaryAction?: (text: string) => void;
  primaryActionLabel?: string;
}) {
  const [activeToneId, setActiveToneId] = useState(tones[0]?.id);
  return (
    <DraftCard
      title={title}
      tones={tones}
      activeToneId={activeToneId}
      onSelectTone={setActiveToneId}
      onCopy={onCopy}
      onPrimaryAction={onPrimaryAction}
      primaryActionLabel={primaryActionLabel}
    />
  );
}

function InteractiveChoicePrompt({
  question,
  hint,
  options,
  onSelectChoice,
}: {
  question: string;
  hint?: string;
  options: ChoiceOption[];
  onSelectChoice?: (question: string, choiceId: string) => void;
}) {
  const [selectedId, setSelectedId] = useState<string | undefined>(undefined);
  return (
    <ChoicePromptCard
      question={question}
      hint={hint}
      options={options}
      selectedId={selectedId}
      onSelect={(id) => {
        setSelectedId(id);
        onSelectChoice?.(question, id);
      }}
    />
  );
}

function renderGenUiCard(
  payload: Record<string, unknown>,
  raw: string,
  idx: number,
  props: GenUiRendererProps,
) {
  const kind = String(payload.type ?? '');
  const copyRaw = () => props.onCopyText?.(raw);
  const saveRaw = () => props.onSaveText?.(raw);

  switch (kind) {
    case 'compare_table': {
      const rows = Array.isArray(payload.rows) ? (payload.rows as CompareTableRow[]) : [];
      return (
        <CompareTable
          key={`g-${idx}`}
          title={String(payload.title ?? 'Comparison')}
          badgeText={typeof payload.badgeText === 'string' ? payload.badgeText : undefined}
          leftColumnTitle={String(payload.leftColumnTitle ?? 'Option A')}
          rightColumnTitle={String(payload.rightColumnTitle ?? 'Option B')}
          rows={rows}
          footerLabel={typeof payload.footerLabel === 'string' ? payload.footerLabel : undefined}
          onCopy={copyRaw}
          onSave={saveRaw}
        />
      );
    }
    case 'checklist': {
      const items = Array.isArray(payload.items) ? (payload.items as ChecklistItem[]) : [];
      return (
        <InteractiveChecklist
          key={`g-${idx}`}
          title={String(payload.title ?? 'Checklist')}
          initialItems={items}
          onCopy={copyRaw}
        />
      );
    }
    case 'page_brief': {
      const bullets = Array.isArray(payload.bullets)
        ? payload.bullets.map((b) => String(b))
        : [];
      return (
        <PageBriefCard
          key={`g-${idx}`}
          title={String(payload.title ?? 'Page Brief')}
          subtitle={typeof payload.subtitle === 'string' ? payload.subtitle : undefined}
          originLabel={typeof payload.originLabel === 'string' ? payload.originLabel : undefined}
          bullets={bullets}
          highlightsCount={
            typeof payload.highlightsCount === 'number' ? payload.highlightsCount : undefined
          }
          highlightsContent={
            typeof payload.highlightsContent === 'string' ? payload.highlightsContent : undefined
          }
          onCopy={copyRaw}
        />
      );
    }
    case 'draft': {
      const tones = Array.isArray(payload.tones) ? (payload.tones as DraftToneOption[]) : [];
      return (
        <InteractiveDraft
          key={`g-${idx}`}
          title={typeof payload.title === 'string' ? payload.title : undefined}
          tones={tones}
          onCopy={(c) => props.onCopyText?.(c)}
          onPrimaryAction={(c) => props.onSaveText?.(c)}
          primaryActionLabel={
            typeof payload.primaryActionLabel === 'string' ? payload.primaryActionLabel : undefined
          }
        />
      );
    }
    case 'entity_summary': {
      const fields = Array.isArray(payload.fields) ? (payload.fields as EntityField[]) : [];
      return (
        <EntitySummaryCard
          key={`g-${idx}`}
          title={String(payload.title ?? 'Summary')}
          subtitle={typeof payload.subtitle === 'string' ? payload.subtitle : undefined}
          tag={typeof payload.tag === 'string' ? payload.tag : undefined}
          fields={fields}
          onSave={saveRaw}
        />
      );
    }
    case 'choice_prompt': {
      const options = Array.isArray(payload.options) ? (payload.options as ChoiceOption[]) : [];
      const question = String(payload.question ?? 'Select an option');
      return (
        <InteractiveChoicePrompt
          key={`g-${idx}`}
          question={question}
          hint={typeof payload.hint === 'string' ? payload.hint : undefined}
          options={options}
          onSelectChoice={props.onSelectChoice}
        />
      );
    }
    case 'metric_strip': {
      const values = Array.isArray(payload.values)
        ? payload.values.filter((v): v is number => typeof v === 'number')
        : [];
      return (
        <MetricStripCard
          key={`g-${idx}`}
          label={String(payload.label ?? 'Metrics')}
          averageLabel={typeof payload.averageLabel === 'string' ? payload.averageLabel : undefined}
          primaryValue={String(payload.primaryValue ?? '—')}
          unitLabel={typeof payload.unitLabel === 'string' ? payload.unitLabel : undefined}
          deltaLabel={typeof payload.deltaLabel === 'string' ? payload.deltaLabel : undefined}
          values={values}
          lowLabel={typeof payload.lowLabel === 'string' ? payload.lowLabel : undefined}
          highLabel={typeof payload.highLabel === 'string' ? payload.highLabel : undefined}
          footnote={typeof payload.footnote === 'string' ? payload.footnote : undefined}
        />
      );
    }
    case 'page_action': {
      const fields = Array.isArray(payload.fields) ? (payload.fields as PageActionField[]) : [];
      const title = String(payload.title ?? 'Page Action');
      return (
        <PageActionCard
          key={`g-${idx}`}
          title={title}
          subtitle={typeof payload.subtitle === 'string' ? payload.subtitle : undefined}
          fields={fields}
          onCancel={() => props.onCopyText?.(`Cancelled action: ${title}`)}
          onConfirm={() => props.onConfirmAction?.(title)}
        />
      );
    }
    case 'timeline': {
      const rawSteps = Array.isArray(payload.steps) ? payload.steps : [];
      const steps: TimelineStep[] = rawSteps.map((s, sIdx) => {
        const rec = (s && typeof s === 'object' ? s : {}) as Record<string, unknown>;
        const st = String(rec.status ?? 'done');
        const status: TimelineStep['status'] =
          st === 'done' || st === 'active' || st === 'pending' || st === 'error' ? st : 'done';
        return {
          id: typeof rec.id === 'string' && rec.id ? rec.id : `step-${sIdx}`,
          label: String(rec.label ?? rec.title ?? `Step ${sIdx + 1}`),
          detail: typeof rec.detail === 'string' ? rec.detail : undefined,
          timestamp: typeof rec.timestamp === 'string' ? rec.timestamp : undefined,
          status,
        };
      });
      return (
        <TimelineCard
          key={`g-${idx}`}
          title={String(payload.title ?? 'Timeline')}
          subtitle={typeof payload.subtitle === 'string' ? payload.subtitle : undefined}
          metaLabel={typeof payload.metaLabel === 'string' ? payload.metaLabel : undefined}
          steps={steps}
        />
      );
    }
    case 'quote_excerpt': {
      return (
        <QuoteExcerptCard
          key={`g-${idx}`}
          quote={String(payload.quote ?? '')}
          sourceTitle={String(payload.sourceTitle ?? payload.title ?? 'Source')}
          sourceDomain={typeof payload.sourceDomain === 'string' ? payload.sourceDomain : undefined}
          sourceUrl={typeof payload.sourceUrl === 'string' ? payload.sourceUrl : undefined}
          locator={typeof payload.locator === 'string' ? payload.locator : undefined}
          onCopyQuote={(q) => props.onCopyText?.(q)}
        />
      );
    }
    default:
      return (
        <CodeBlockCard
          key={`g-${idx}`}
          code={raw.trim()}
          language="json"
          onCopy={(c) => props.onCopyText?.(c)}
        />
      );
  }
}

/**
 * Renders assistant markdown responses safely in Chrome MV3:
 * - Dispatches ` ```genui ` JSON payloads to `@yours/ui/genui` cards.
 * - Renders settled fenced code blocks via `@yours/ui` `CodeBlockCard` (zero inline styles).
 * - Uses `@yours/ui` `StreamingResponse` for progressive optical sharpen/develop reveal,
 *   gliding caret, and native GFM pipe tables.
 * - Zero raw HTML string injection, zero inline CSS style objects.
 */
export function GenUiRenderer(props: GenUiRendererProps) {
  const segments = splitForStreamingRenderer(props.content, props.status);
  if (segments.length === 0) {
    return null;
  }

  return (
    <div className="ext-md-root">
      {segments.map((seg, idx) => {
        if (seg.kind === 'genui') {
          return renderGenUiCard(seg.payload, seg.raw, idx, props);
        }
        if (seg.kind === 'code') {
          return (
            <CodeBlockCard
              key={`cb-${idx}`}
              code={seg.value}
              language={seg.language}
              showLineNumbers
              onCopy={props.onCopyText}
            />
          );
        }
        const isLast = idx === segments.length - 1;
        const textWithCitations =
          props.sources && props.sources.length > 0
            ? linkifyInlineCitations(seg.value, props.sources)
            : seg.value;
        return (
          <StreamingResponse
            key={`sr-${idx}`}
            text={textWithCitations}
            status={isLast && props.status === 'streaming' ? 'streaming' : 'done'}
            speed={props.streamingSpeed ?? 20}
            ink="accent"
            actions={false}
            onCopy={props.onCopyText}
          />
        );
      })}
    </div>
  );
}
