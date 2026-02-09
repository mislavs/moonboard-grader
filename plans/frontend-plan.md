# Moonboard Grader Frontend Plan

## Step 1: Project Setup (Hello World) ✅ COMPLETED

- ✅ Create new Vite + React + TypeScript project in `/frontend` directory
- ✅ Install and configure TailwindCSS
- ✅ Create basic folder structure: `/components`, `/types`, `/utils`, `/assets`
- ⏭️ Add moonboard image to `/assets` (will add when needed in Step 2)
- ✅ Verify app runs with "Hello World"

## Step 2: Display Hardcoded Problem ✅ COMPLETED

- ✅ Create TypeScript types for Problem, Move, etc. based on JSON structure
- ✅ Create `MoonBoard` component:
  - ✅ Display moonboard grid as SVG background
  - ✅ Create absolute positioning overlay for holds
  - ✅ Calculate grid positions (11 cols × 18 rows, A-K, 1-18)
  - ✅ Hardcode a problem JSON in the component (App.tsx)
  - ✅ Render colored circles over holds:
    - ✅ Green circle for start holds (isStart: true)
    - ✅ Red circle for end holds (isEnd: true)
    - ✅ Blue circle for intermediate holds
  - ✅ Display problem name and grade above the board

## Step 3: Problem Creator ✅ COMPLETED

- ✅ Add mode toggle: "View" vs "Create" (implemented as tabs at top of page)
- ✅ In Create mode:
  - ✅ Allow clicking on moonboard grid positions to add/remove holds
  - ✅ Click once = intermediate hold (blue)
  - ✅ Click again = start hold (green)
  - ✅ Click again = end hold (red)
  - ✅ Click again = remove hold
- ✅ Add controls:
  - ✅ "Clear All" button to reset
  - ✅ "Predict Grade" button (initially just logs the moves)
  - ✅ Display current moves array as JSON for debugging

## Step 4: Backend Integration for Predictions ✅ COMPLETED

- ✅ Create API service (`/services/api.ts`) to call FastAPI backend at `http://localhost:8000`
- ✅ Update "Predict Grade" button to:
  - ✅ Send moves to `/predict` endpoint
  - ✅ Handle loading state
  - ✅ Handle errors (backend not running, etc.)
- ✅ Create `PredictionDisplay` component:
  - ✅ Show predicted grade prominently
  - ✅ Display confidence percentage
  - ✅ Show top-k predictions in a list with probabilities
- ✅ Add API health check on app load to verify backend connection

## Step 5: Problem Navigator ✅ COMPLETED

- ✅ Create `ProblemNavigator` component:
  - ✅ Display list of problems (name + grade)
  - ✅ Add pagination controls for navigating through problem list
  - ✅ Highlight selected problem
- ✅ Click to select a problem
- ✅ Update app state to show selected problem on MoonBoard
- ✅ Create responsive 2-column layout: Navigator | MoonBoard