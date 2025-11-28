import type { Move } from "../types/problem";
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
}

interface BoardCanvasCreateProps {
  moves: Move[];
  mode: "create";
  onMovesChange: (moves: Move[]) => void;
  attentionMap?: number[][];
  showAttention?: boolean;
}

type BoardCanvasProps = BoardCanvasViewProps | BoardCanvasCreateProps;

export default function BoardCanvas(props: BoardCanvasProps) {
  const {
    moves,
    mode,
    onMovesChange,
    attentionMap,
    showAttention = false,
  } = props;
  const { width, height, holdRadius } = BOARD_CONFIG;

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

          return (
            <circle
              key={`${move.description}-${index}`}
              cx={x}
              cy={y}
              r={holdRadius}
              fill={mode === "create" ? "transparent" : "none"}
              stroke={color}
              strokeWidth={5}
              style={mode === "create" ? { cursor: "pointer" } : undefined}
            />
          );
        })}
      </svg>
    </div>
  );
}
