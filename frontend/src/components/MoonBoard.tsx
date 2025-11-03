import type { Problem } from '../types/problem';
import { parsePosition, gridToPixel } from '../utils/gridParser';
import moonboardImage from '../assets/moonboard.jpg';

interface MoonBoardProps {
  problem: Problem;
}

export default function MoonBoard({ problem }: MoonBoardProps) {
  const boardWidth = 550;
  const boardHeight = 900;
  const holdRadius = 22;

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Problem Info */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-2">{problem.name}</h2>
        <div className="flex items-center justify-center gap-4 text-gray-300">
          <span className="text-2xl font-semibold text-blue-400">{problem.grade}</span>
          <span>•</span>
          <span>by {problem.setby}</span>
          <span>•</span>
          <span>{problem.repeats} repeats</span>
        </div>
      </div>

      {/* Moonboard */}
      <div className="relative bg-gray-800 rounded-lg shadow-2xl p-4">
        {/* Grid background with image */}
        <div className="relative border-4 border-gray-700 rounded overflow-hidden">
          {/* Background Image */}
          <img 
            src={moonboardImage} 
            alt="Moonboard" 
            className="absolute inset-0 w-full h-full object-cover"
            style={{ width: boardWidth, height: boardHeight }}
          />
          
          {/* SVG Overlay for holds */}
          <svg
            width={boardWidth}
            height={boardHeight}
            className="relative"
            style={{ display: 'block' }}
          >
            {/* Holds */}
            {problem.moves.map((move, index) => {
              const position = parsePosition(move.description);
              const { x, y } = gridToPixel(position, boardWidth, boardHeight);

              // Determine color based on hold type
              let color = '#3b82f6'; // blue for intermediate
              if (move.isStart) {
                color = '#22c55e'; // green for start
              } else if (move.isEnd) {
                color = '#ef4444'; // red for end
              }

              return (
                <circle
                  key={`${move.description}-${index}`}
                  cx={x}
                  cy={y}
                  r={holdRadius}
                  fill="none"
                  stroke={color}
                  strokeWidth={5}
                />
              );
            })}
          </svg>
        </div>

        {/* Legend */}
        <div className="flex justify-center gap-6 mt-4">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-green-500"></div>
            <span className="text-sm text-gray-300">Start</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-500"></div>
            <span className="text-sm text-gray-300">Intermediate</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-red-500"></div>
            <span className="text-sm text-gray-300">End</span>
          </div>
        </div>
      </div>

      {/* Additional Info */}
      <div className="bg-gray-800 rounded-lg p-4 max-w-md">
        <div className="text-gray-300 space-y-2">
          <p><span className="font-semibold text-white">Setup:</span> {problem.holdsetup.description}</p>
        </div>
      </div>
    </div>
  );
}

