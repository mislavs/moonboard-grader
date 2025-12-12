import type { Problem, Move } from "../types/problem";
import type { BetaResponse } from "../types/beta";
import BoardCanvas from "./BoardCanvas";
import BoardLegend from "./BoardLegend";

interface MoonBoardViewProps {
  problem: Problem;
  mode: "view";
  attentionMap?: number[][];
  showAttention?: boolean;
  beta?: BetaResponse;
  showBeta?: boolean;
}

interface MoonBoardCreateProps {
  moves: Move[];
  mode: "create";
  onMovesChange: (moves: Move[]) => void;
  attentionMap?: number[][];
  showAttention?: boolean;
  beta?: BetaResponse;
  showBeta?: boolean;
}

type MoonBoardProps = MoonBoardViewProps | MoonBoardCreateProps;

export default function MoonBoard(props: MoonBoardProps) {
  const { attentionMap, showAttention, beta, showBeta } = props;

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
            beta={beta}
            showBeta={showBeta}
          />
        ) : (
          <BoardCanvas
            moves={props.moves}
            mode="create"
            onMovesChange={props.onMovesChange}
            attentionMap={attentionMap}
            showAttention={showAttention}
            beta={beta}
            showBeta={showBeta}
          />
        )}
        <BoardLegend />
      </div>
    </div>
  );
}
