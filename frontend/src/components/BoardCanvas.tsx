import type { Move } from '../types/problem';
import { parsePosition, gridToPixel } from '../utils/gridParser';
import { getHoldColor } from '../utils/holdUtils';
import { BOARD_CONFIG } from '../constants/boardConfig';
import moonboardImage from '../assets/moonboard.jpg';

interface BoardCanvasProps {
  moves: Move[];
}

export default function BoardCanvas({ moves }: BoardCanvasProps) {
  const { width, height, holdRadius } = BOARD_CONFIG;

  return (
    <div className="relative border-4 border-gray-700 rounded overflow-hidden">
      {/* Background Image */}
      <img 
        src={moonboardImage} 
        alt="Moonboard" 
        className="absolute inset-0 w-full h-full object-cover"
        style={{ width, height }}
      />
      
      {/* SVG Overlay for holds */}
      <svg
        width={width}
        height={height}
        className="relative"
        style={{ display: 'block' }}
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
              fill="none"
              stroke={color}
              strokeWidth={5}
            />
          );
        })}
      </svg>
    </div>
  );
}

