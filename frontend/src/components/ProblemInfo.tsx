import type { Problem } from '../types/problem';

interface ProblemInfoProps {
  problem: Problem;
}

export default function ProblemInfo({ problem }: ProblemInfoProps) {
  return (
    <>
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-2">
          {problem.name}
        </h2>
        <div className="flex items-center justify-center gap-4 text-gray-300">
          <span className="text-2xl font-semibold text-blue-400">
            {problem.grade}
          </span>
          <span>•</span>
          <span>by {problem.setby}</span>
          <span>•</span>
          <span>{problem.repeats} repeats</span>
        </div>
      </div>
    </>
  );
}

