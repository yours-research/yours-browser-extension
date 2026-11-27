import { useEffect, useRef, useState } from 'react';
import { ChevronLeftIcon, ChevronRightIcon, ExternalLinkIcon, GlobeIcon, XIcon } from '@yours/ui';
import type { TurnResponseSource } from '../responseSources.ts';

export interface TurnSourcesTriggerProps {
  sources: readonly TurnResponseSource[];
  onOpenUrl: (url: string) => void;
}

/**
 * Compact per-turn source trigger (`Sources (N)`) paired beside an assistant message's
 * ActionBar, opening a clean single popover preview with previous/next switching and
 * external URL navigation (never "Show on page" for external web URLs).
 */
export function TurnSourcesTrigger({ sources, onOpenUrl }: TurnSourcesTriggerProps) {
  const [open, setOpen] = useState(false);
  const [activeIdx, setActiveIdx] = useState(0);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (activeIdx >= sources.length) {
      setActiveIdx(0);
    }
  }, [sources.length, activeIdx]);

  useEffect(() => {
    if (!open) return;
    const onDocDown = (ev: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(ev.target as Node)) {
        setOpen(false);
      }
    };
    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDocDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onDocDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  if (sources.length === 0) {
    return null;
  }

  const current = sources[Math.min(activeIdx, sources.length - 1)]!;
  const safeCount = sources.length;

  return (
    <div ref={rootRef} className="ext-turn-sources" data-slot="turn-sources">
      <button
        type="button"
        className={`yui-chip-btn ext-sources-pill ${open ? 'is-active' : ''}`}
        data-slot="turn-sources-pill"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((prev) => !prev)}
      >
        <GlobeIcon size={12} />
        <span>Sources ({safeCount})</span>
      </button>

      {open ? (
        <div
          className="ext-sources-popover"
          role="dialog"
          aria-label="Turn web sources"
          data-slot="turn-sources-popover"
        >
          <div className="ext-sources-popover-head">
            <span className="ext-sources-popover-domain">{current.domain}</span>
            <div className="ext-sources-popover-nav">
              {safeCount > 1 ? (
                <>
                  <button
                    type="button"
                    className="yui-icon-btn"
                    aria-label="Previous source"
                    onClick={() => setActiveIdx((i) => (i - 1 + safeCount) % safeCount)}
                  >
                    <ChevronLeftIcon size={12} />
                  </button>
                  <span className="ext-sources-popover-count">
                    {activeIdx + 1} / {safeCount}
                  </span>
                  <button
                    type="button"
                    className="yui-icon-btn"
                    aria-label="Next source"
                    onClick={() => setActiveIdx((i) => (i + 1) % safeCount)}
                  >
                    <ChevronRightIcon size={12} />
                  </button>
                </>
              ) : null}
              <button
                type="button"
                className="yui-icon-btn"
                aria-label="Close sources popover"
                onClick={() => setOpen(false)}
              >
                <XIcon size={12} />
              </button>
            </div>
          </div>

          <div className="ext-sources-popover-title">{current.title}</div>
          {current.snippet ? (
            <p className="ext-sources-popover-snippet">{current.snippet}</p>
          ) : null}

          <div className="ext-sources-popover-foot">
            <button
              type="button"
              className="yui-chip-btn ext-sources-open-btn"
              onClick={() => onOpenUrl(current.url)}
            >
              <ExternalLinkIcon size={12} />
              <span>Open source</span>
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
