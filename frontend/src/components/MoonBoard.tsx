import type { Problem, Move } from '../types/problem';
import BoardCanvas from './BoardCanvas';
import BoardLegend from './BoardLegend';
import ProblemInfo from './ProblemInfo';

interface MoonBoardViewProps {
  problem: Problem;
  mode: 'view';
}

interface MoonBoardCreateProps {
  moves: Move[];
  mode: 'create';
  onMovesChange: (moves: Move[]) => void;
}

type MoonBoardProps = MoonBoardViewProps | MoonBoardCreateProps;

export default function MoonBoard(props: MoonBoardProps) {
  return (
    <div className="flex flex-col items-center gap-6">
      {props.mode === 'view' && <ProblemInfo problem={props.problem} />}

      {/* Moonboard */}
      <div className="relative bg-gray-800 rounded-lg shadow-2xl p-4">
        {props.mode === 'view' ? (
          <BoardCanvas moves={props.problem.moves} mode="view" />
        ) : (
          <BoardCanvas
            moves={props.moves}
            mode="create"
            onMovesChange={props.onMovesChange}
          />
        )}
        <BoardLegend />
      </div>
    </div>
  );
}

