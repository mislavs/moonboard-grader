# Moonboard Grader Frontend Plan

## Step 1: Project Setup (Hello World)

- Create new Vite + React + TypeScript project in `/frontend` directory
- Install and configure TailwindCSS
- Create basic folder structure: `/components`, `/types`, `/utils`, `/assets`
- Add moonboard image to `/assets`
- Verify app runs with "Hello World"

## Step 2: Display Hardcoded Problem

- Create TypeScript types for Problem, Move, etc. based on JSON structure
- Create `MoonBoard` component:
- Display moonboard image as background
- Create absolute positioning overlay for holds
- Calculate grid positions (11 cols × 18 rows, A-K, 1-18)
- Hardcode a problem JSON in the component
- Render colored circles over holds:
- Green circle for start holds (isStart: true)
- Red circle for end holds (isEnd: true)
- Blue circle for intermediate holds
- Display problem name and grade above the board

## Step 3: Problem Creator

- Add mode toggle: "View" vs "Create"
- In Create mode:
- Allow clicking on moonboard grid positions to add/remove holds
- Click once = intermediate hold (blue)
- Click again = start hold (green)
- Click again = end hold (red)
- Click again = remove hold
- Add controls:
- "Clear All" button to reset
- "Predict Grade" button (initially just logs the moves)
- Display current moves array as JSON for debugging

## Step 4: Backend Integration for Predictions

- Create API service (`/utils/api.ts`) to call FastAPI backend at `http://localhost:8000`
- Update "Predict Grade" button to:
- Send moves to `/predict` endpoint
- Handle loading state
- Handle errors (backend not running, etc.)
- Create `PredictionDisplay` component:
- Show predicted grade prominently
- Display confidence percentage
- Show top-k predictions in a list with probabilities
- Add API health check on app load to verify backend connection

## Step 5: Problem Navigator

- Copy the full problems JSON file to `/frontend/src/data/problems.json`
- Create `ProblemNavigator` component:
- Display list of problems (name + grade)
- Search/filter by name or grade
- Click to select a problem
- Update app state to show selected problem on MoonBoard
- Add problem details panel showing: name, grade, setter, repeats, method