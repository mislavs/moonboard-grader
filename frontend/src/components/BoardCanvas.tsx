import type { Move } from "../types/problem";
import type { BetaResponse } from "../types/beta";
import {
  parsePosition,
  gridToPixel,
  pixelToGrid,
  formatPosition,
} from "../utils/gridParser";
import { getHoldColor } from "../utils/holdUtils";
import { getNextMoveState } from "../utils/moveStateManager";
import { BOARD_CONFIG } from "../config/board";
import moonboardImage from "../assets/moonboard.jpg";
import AttentionHeatmap from "./AttentionHeatmap";

interface BoardCanvasViewProps {
  moves: Move[];
  mode: "view";
  onMovesChange?: never;
  attentionMap?: number[][];
  showAttention?: boolean;
  beta?: BetaResponse;
  showBeta?: boolean;
}

interface BoardCanvasCreateProps {
  moves: Move[];
  mode: "create";
  onMovesChange: (moves: Move[]) => void;
  attentionMap?: number[][];
  showAttention?: boolean;
  beta?: BetaResponse;
  showBeta?: boolean;
}

type BoardCanvasProps = BoardCanvasViewProps | BoardCanvasCreateProps;

export default function BoardCanvas(props: BoardCanvasProps) {
  const {
    moves,
    mode,
    onMovesChange,
    attentionMap,
    showAttention = false,
    beta,
    showBeta = false,
  } = props;
  const { width, height, holdRadius } = BOARD_CONFIG;

  // Build lookup map from hold description to hand assignments (can have both hands on same hold)
  const holdToHands = new Map<string, Set<"LH" | "RH">>();
  if (beta && showBeta) {
    // Check if there's only one start hold - if so, both hands start there
    const startHolds = moves.filter(m => m.isStart);
    const hasSingleStartHold = startHolds.length === 1;

    if (hasSingleStartHold) {
      // Single start hold gets both hands
      holdToHands.set(startHolds[0].description, new Set(["LH", "RH"]));
    } else {
      // Multiple start holds - use beta response
      for (const startHand of beta.startHands) {
        if (!holdToHands.has(startHand.description)) {
          holdToHands.set(startHand.description, new Set());
        }
        holdToHands.get(startHand.description)!.add(startHand.hand);
      }
    }

    // Add moves (may override starts for holds used again)
    for (const betaMove of beta.moves) {
      holdToHands.set(betaMove.hold, new Set([betaMove.hand]));
    }
  }

  const handleClick = (event: React.MouseEvent<SVGSVGElement>) => {
    if (mode !== "create" || !onMovesChange) return;

    const svg = event.currentTarget;
    const rect = svg.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const gridPos = pixelToGrid(x, y, width, height);
    if (!gridPos) return;

    const description = formatPosition(gridPos);
    const newMoves = getNextMoveState(moves, description);
    onMovesChange(newMoves);
  };

  return (
    <div className="relative border-4 border-gray-700 rounded overflow-hidden">
      {/* Background Image */}
      <img
        src={moonboardImage}
        alt="Moonboard"
        className="absolute inset-0 w-full h-full object-cover"
        style={{ width, height }}
      />

      {/* Attention Heatmap Overlay (between background and holds) */}
      {showAttention && attentionMap && (
        <AttentionHeatmap attentionMap={attentionMap} />
      )}

      {/* SVG Overlay for holds */}
      <svg
        width={width}
        height={height}
        className={`relative ${mode === "create" ? "cursor-crosshair" : ""}`}
        style={{ display: "block", zIndex: 2 }}
        onClick={handleClick}
      >
        {moves.map((move, index) => {
          const position = parsePosition(move.description);
          const { x, y } = gridToPixel(position, width, height);
          const color = getHoldColor(move);
          const hands = holdToHands.get(move.description);
          const handsArray = hands ? Array.from(hands) : [];
          const hasBothHands = handsArray.length === 2;

          return (
            <g key={`${move.description}-${index}`}>
              <circle
                cx={x}
                cy={y}
                r={holdRadius}
                fill={mode === "create" ? "transparent" : "none"}
                stroke={color}
                strokeWidth={5}
                style={mode === "create" ? { cursor: "pointer" } : undefined}
              />
              {showBeta && handsArray.length > 0 && (
                hasBothHands ? (
                  // Both hands on same hold - show side by side
                  <>
                    <text
                      x={x - 10}
                      y={y - holdRadius - 8}
                      textAnchor="middle"
                      fontSize="14"
                      fontWeight="bold"
                      fill="#60a5fa"
                      stroke="#000"
                      strokeWidth="0.5"
                    >
                      L
                    </text>
                    <text
                      x={x + 10}
                      y={y - holdRadius - 8}
                      textAnchor="middle"
                      fontSize="14"
                      fontWeight="bold"
                      fill="#fb923c"
                      stroke="#000"
                      strokeWidth="0.5"
                    >
                      R
                    </text>
                  </>
                ) : (
                  // Single hand
                  <text
                    x={x}
                    y={y - holdRadius - 8}
                    textAnchor="middle"
                    fontSize="14"
                    fontWeight="bold"
                    fill={handsArray[0] === "LH" ? "#60a5fa" : "#fb923c"}
                    stroke="#000"
                    strokeWidth="0.5"
                  >
                    {handsArray[0] === "LH" ? "L" : "R"}
                  </text>
                )
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
