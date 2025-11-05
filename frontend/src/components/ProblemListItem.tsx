import type { ProblemListItem as ProblemData } from '../services/api';

interface ProblemListItemProps {
  problem: ProblemData;
  isSelected: boolean;
  onSelect: (id: number) => void;
}

export default function ProblemListItem({
  problem,
  isSelected,
  onSelect,
}: ProblemListItemProps) {
  return (
    <button
      onClick={() => onSelect(problem.id)}
      className={`w-full text-left p-3 rounded transition-colors ${
        isSelected
          ? 'bg-blue-600 text-white'
          : 'bg-gray-700 text-gray-200 hover:bg-gray-600'
      }`}
    >
      <div className="font-semibold">{problem.name}</div>
      <div className="text-sm opacity-80">Grade: {problem.grade}</div>
    </button>
  );
}

