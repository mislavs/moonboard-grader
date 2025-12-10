# Implementation Plan: Dynamic Programming Beta Sequence Generator

## Progress

- [x] **Step 1**: Define Data Models
- [ ] **Step 2**: Implement Scoring Function
- [ ] **Step 3**: Implement DP Core Algorithm
- [ ] **Step 4**: Implement Entry Point (CLI)

## Overview

**Goal**: Build an algorithm that determines the optimal hand sequence (beta) for completing a MoonBoard climbing problem. Given a set of holds with positions and start/end markers, the algorithm outputs which hand (left or right) should grab each hold and in what order.

**Context**:
- **MoonBoard**: A standardized indoor climbing training wall with a fixed grid of holds (11 columns labeled A-K, 18 rows labeled 1-18)
- **Beta**: Climbing terminology for the sequence of movements to complete a route
- **Problem**: A specific route consists of designated start holds, intermediate holds, and finish holds

**Architecture**: Two decoupled components:
1. **DP Search Algorithm**: Finds the optimal sequence by exhaustively exploring states with memoization
2. **Scoring Function**: Evaluates how "good" a particular move is (pluggable, can be improved independently)

---

## Step 1: Define Data Models ✅

### 1.1 Hold Representation
Create a Hold class/struct/record with:
- **x**: Column position (0-10, where 0=A, 10=K)
- **y**: Row position (0-17, where 0=row 1, 17=row 18)
- **is_start**: Boolean, true if this is a starting hold
- **is_end**: Boolean, true if this is a finishing hold

### 1.2 Move Representation
Create a Move class/struct/record with:
- **hold_index**: Which hold is being grabbed (index into input list)
- **hand**: Which hand grabs it ("LH" for left hand, "RH" for right hand)

### 1.3 Problem Representation
A Problem is a list of Holds, typically 5-15 holds total, sorted by y-coordinate (bottom to top). A valid problem has:
- 1-2 start holds (at the bottom)
- 1-2 end holds (at the top)
- 0 or more intermediate holds

### 1.4 Input Format (JSON)
The input comes from `data/problems_100.json` with the following structure:

```json
{
  "total": 100,
  "data": [
    {
      "name": "Problem Name",
      "grade": "6B+",
      "moves": [
        {
          "description": "J4",
          "isStart": true,
          "isEnd": false
        },
        {
          "description": "G6",
          "isStart": false,
          "isEnd": false
        },
        {
          "description": "F18",
          "isStart": false,
          "isEnd": true
        }
      ],
      "apiId": 305445
    }
  ]
}
```

**Parsing requirements:**
- Parse the `description` field (e.g., "J4") into x, y coordinates:
  - Column letter A-K → x value 0-10 (A=0, B=1, ..., K=10)
  - Row number 1-18 → y value 0-17 (row 1=0, row 18=17)
- Map `isStart` → `is_start` and `isEnd` → `is_end`
- Sort holds by y-coordinate (bottom to top) after parsing

---

## Step 2: Implement Scoring Function

### 2.1 Design Principle
Start with a simple, interpretable scoring function. This can be refined later without changing the DP algorithm.

### 2.2 Function Signature
```
ScoreMove(targetIndex, remainingIndex, hand, holds) → float

Parameters:
  - targetIndex: the hold being grabbed (int)
  - remainingIndex: the hold the other hand is on (int)
  - hand: "LH" or "RH" (enum)
  - holds: the full list of holds (for coordinate lookup)

Returns:
  - A single numeric score (use log-scale internally)
```

### 2.3 Score Components

**Component 1: Distance Penalty**
- Penalize moves where the target hold is far from the stationary (remaining) hand
- Use exponential decay: higher score for closer targets
- Suggested scaling: comfortable reach is ~3-4 grid units; beyond 6 units is very difficult

**Component 2: Cross-Body Penalty**
- Penalize moves where hands cross each other
- If moving RH to a position significantly left of LH: apply penalty
- If moving LH to a position significantly right of RH: apply penalty
- Suggested threshold: penalize if crossing by more than 1 grid unit

**Component 3: Upward Movement Bonus**
- Slightly favor moves that progress upward
- Small bonus when target y-coordinate is higher than current

### 2.4 Score Combination
Multiply all component scores together. Each component should return a value between 0 and 1 (or slightly above 1 for bonuses).

### 2.5 Numerical Stability
Convert multiplicative scores to additive by taking logarithms:
- Store log(score) instead of score
- Sum log-scores instead of multiplying scores
- This prevents numerical underflow with many moves

### 2.6 Tie-breaking

- prefer moves with lower distance penalty
- if distance penalty is the same, consider the moves as same difficulty and return the first one calculated

### 2.7 Tests
- Test distance penalty: farther targets should score lower
- Test cross-body penalty: crossing moves should score lower
- Test that scores are in expected range (0 to ~1)

---

## Step 3: Implement DP Core Algorithm

### 3.1 State Definition
A state is a tuple of three elements:
- **lh_position**: Index of the hold currently held by left hand
- **rh_position**: Index of the hold currently held by right hand
- **visited_mask**: Bitmask integer where bit i is set if hold i has been visited

### 3.2 Implement Initialization Logic
1. Identify start holds from the input
2. If 2 start holds: Try both LH/RH assignments, keep the better result
3. If 1 start hold: Both hands start matched on the same hold
4. Initial visited_mask has bits set for all start holds

### 3.3 Implement Transitions
From any state, valid transitions are:
- Move LH to any unvisited hold → new state with updated lh_position and visited_mask
- Move RH to any unvisited hold → new state with updated rh_position and visited_mask

### 3.4 Implement Termination Condition
A state is terminal when all holds have been visited (visited_mask has all N bits set).

### 3.5 Implement Recurrence with Memoization
```
optimal_score(lh, rh, visited) = 
    if terminal: return 0
    else: return max over all valid transitions of:
        score(transition) + optimal_score(new_state)
```

Store computed results in a hash map/dictionary keyed by (lh, rh, visited) to avoid recomputation.

### 3.6 Implement Path Reconstruction
Along with the score, store the sequence of moves that achieved that score. When memoizing, store both the best score and the corresponding move sequence.

### 3.7 End Hold Handling
End holds should be visited like any other hold. No special handling needed in the DP logic itself—the scoring function can optionally give bonuses/penalties related to finishing.

### 3.8 Tests
- Test with trivial 3-hold problem (start, middle, end): should find obvious sequence
- Test with 4-hold problem requiring hand alternation

---

## Step 4: Implement Entry Point (CLI)

1. Parse/validate input
2. Determine start configuration(s)
3. Call DP with initial state(s)
4. If multiple start configurations, run both and return the better one
5. Format and return output

### 4.1 Output Format
An ordered list of moves. Example for a 7-hold problem:
```
LH start: Hold E5
RH start: Hold E5

Move 1: Hold I8, RH
Move 2: Hold H10, LH
Move 3: Hold G13, RH
Move 4: Hold E15, LH
Move 5: Hold F18, RH (finish)
```

Also include:
- Total score of the sequence
- Score breakdown per move for debugging

---

## Future Improvements (Out of Scope for V1)

These are NOT part of the initial implementation but document future enhancement paths:

- **Scoring V2**: Add hold difficulty ratings (single value per hold)
- **Scoring V3**: Add hand-specific difficulty ratings
- **Scoring V4**: Add foot placement considerations

