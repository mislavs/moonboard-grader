import type { Problem } from '../types/problem';
import BoardCanvas from './BoardCanvas';
import BoardLegend from './BoardLegend';
import ProblemInfo from './ProblemInfo';

interface MoonBoardProps {
  problem: Problem;
}

export default function MoonBoard({ problem }: MoonBoardProps) {
  return (
    <div className="flex flex-col items-center gap-6">
      <ProblemInfo problem={problem} />

      {/* Moonboard */}
      <div className="relative bg-gray-800 rounded-lg shadow-2xl p-4">
        <BoardCanvas moves={problem.moves} />
        <BoardLegend />
      </div>
    </div>
  );
}

