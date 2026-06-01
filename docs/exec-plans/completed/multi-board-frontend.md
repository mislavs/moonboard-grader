# Multi-Board Support: Frontend

Implement the frontend changes for multi-board and multi-angle support: dynamic board images and conditional beta toggle.

Depends on the backend plan (`multi-board-backend.md`) for the new fields in the `/board-setups` API response, but can be deployed independently -- missing fields are treated as defaults (`betaSolvingSupported` = true, `boardImage` = null/fallback).

---

## Step 1: Types

**`frontend/src/types/boardSetup.ts`** -- add to `HoldSetup`:
- `betaSolvingSupported: boolean`
- `boardImage: string | null`

Add to `AngleConfig`:
- `hasGenerator: boolean`
- `hasAnalytics: boolean`

## Step 2: Dynamic Board Image

**`frontend/src/components/BoardCanvas.tsx`** and **`frontend/src/components/AnalyticsMode.tsx`**:
- Accept an optional `boardImage` prop
- If set, use `/boards/${boardImage}` as the image source
- If null/undefined, fall back to the existing bundled `moonboard.jpg` import
- The `BoardSetupContext` already exposes `currentHoldSetup` which will have `boardImage`

Board images go in `frontend/public/boards/` (user provides these manually).

## Step 3: Conditional Beta Toggle

**`frontend/src/components/ViewMode.tsx`** and **`frontend/src/components/CreateMode.tsx`**:
- Read `currentHoldSetup.betaSolvingSupported` from context
- Only render `BetaToggle` when `betaSolvingSupported` is true
- Reset beta state when switching to a setup that doesn't support it
- Run `npm run lint` to verify no issues
