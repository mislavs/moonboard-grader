import type { Problem, Move } from "../types/problem";
import BoardCanvas from "./BoardCanvas";
import BoardLegend from "./BoardLegend";

interface MoonBoardViewProps {
  problem: Problem;
  mode: "view";
  attentionMap?: number[][];
  showAttention?: boolean;
}

interface MoonBoardCreateProps {
  moves: Move[];
  mode: "create";
  onMovesChange: (moves: Move[]) => void;
  attentionMap?: number[][];
  showAttention?: boolean;
}

type MoonBoardProps = MoonBoardViewProps | MoonBoardCreateProps;

export default function MoonBoard(props: MoonBoardProps) {
  const { attentionMap, showAttention } = props;

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Moonboard */}
      <div className="relative bg-gray-800 rounded-lg shadow-2xl p-4">
        {props.mode === "view" ? (
          <BoardCanvas
            moves={props.problem.moves}
            mode="view"
            attentionMap={attentionMap}
            showAttention={showAttention}
          />
        ) : (
          <BoardCanvas
            moves={props.moves}
            mode="create"
            onMovesChange={props.onMovesChange}
            attentionMap={attentionMap}
            showAttention={showAttention}
          />
        )}
        <BoardLegend />
      </div>
    </div>
  );
}
