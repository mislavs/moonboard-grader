import { useMemo, useState } from 'react';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import { FONT_GRADES } from '../constants/grades';

interface GradeFilterProps {
  gradeFrom: string | null;
  gradeTo: string | null;
  onGradeFromChange: (grade: string | null) => void;
  onGradeToChange: (grade: string | null) => void;
}

export default function GradeFilter({
  gradeFrom,
  gradeTo,
  onGradeFromChange,
  onGradeToChange,
}: GradeFilterProps) {
  // Convert grade strings to slider indices
  const gradeFromIndex = useMemo(() => {
    return gradeFrom ? FONT_GRADES.indexOf(gradeFrom as typeof FONT_GRADES[number]) : 0;
  }, [gradeFrom]);

  const gradeToIndex = useMemo(() => {
    return gradeTo ? FONT_GRADES.indexOf(gradeTo as typeof FONT_GRADES[number]) : FONT_GRADES.length - 1;
  }, [gradeTo]);

  // Local state for visual updates while dragging
  const [localRange, setLocalRange] = useState<[number, number] | null>(null);

  const handleSliderChange = (value: number | number[]) => {
    if (Array.isArray(value)) {
      setLocalRange([value[0], value[1]]);
    }
  };

  const handleSliderChangeComplete = (value: number | number[]) => {
    if (Array.isArray(value)) {
      const [minIndex, maxIndex] = value;
      
      // Update from grade (null if at minimum)
      const newGradeFrom = minIndex === 0 ? null : FONT_GRADES[minIndex];
      if (newGradeFrom !== gradeFrom) {
        onGradeFromChange(newGradeFrom);
      }
      
      // Update to grade (null if at maximum)
      const newGradeTo = maxIndex === FONT_GRADES.length - 1 ? null : FONT_GRADES[maxIndex];
      if (newGradeTo !== gradeTo) {
        onGradeToChange(newGradeTo);
      }

      // Clear local state
      setLocalRange(null);
    }
  };

  const currentRange = localRange || [gradeFromIndex, gradeToIndex];
  const displayFromGrade = FONT_GRADES[currentRange[0]];
  const displayToGrade = FONT_GRADES[currentRange[1]];

  return (
    <div className="mb-4 pb-4 border-b border-gray-700">
      <h3 className="text-gray-300 text-sm font-semibold mb-2">Grade Range</h3>
      <div className="px-2 py-4">
        <div className="mb-6">
          <Slider
            range
            min={0}
            max={FONT_GRADES.length - 1}
            value={currentRange}
            onChange={handleSliderChange}
            onChangeComplete={handleSliderChangeComplete}
            styles={{
              track: { backgroundColor: '#3b82f6', height: 6 },
              tracks: { backgroundColor: '#3b82f6', height: 6 },
              rail: { backgroundColor: '#4b5563', height: 6 },
              handle: {
                backgroundColor: '#3b82f6',
                borderColor: '#3b82f6',
                opacity: 1,
                width: 18,
                height: 18,
                marginTop: -6,
              },
            }}
          />
        </div>
        <div className="flex justify-between text-sm">
          <div className="text-gray-300">
            <span className="text-gray-400 text-xs">From: </span>
            <span className="font-semibold">{displayFromGrade}</span>
          </div>
          <div className="text-gray-300">
            <span className="text-gray-400 text-xs">To: </span>
            <span className="font-semibold">{displayToGrade}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

