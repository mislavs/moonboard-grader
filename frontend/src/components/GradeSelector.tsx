import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import { AVAILABLE_GRADES } from '../constants/grades';

interface GradeSelectorProps {
  selectedGrade: string;
  onGradeChange: (grade: string) => void;
}

export default function GradeSelector({ selectedGrade, onGradeChange }: GradeSelectorProps) {
  const gradeIndex = AVAILABLE_GRADES.indexOf(selectedGrade as typeof AVAILABLE_GRADES[number]);
  
  const handleSliderChange = (value: number | number[]) => {
    const index = Array.isArray(value) ? value[0] : value;
    onGradeChange(AVAILABLE_GRADES[index]);
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-white font-semibold">Grade</h3>
        <span className="text-purple-400 font-bold text-lg">{selectedGrade}</span>
      </div>
      <div className="pb-2">
        <Slider
          min={0}
          max={AVAILABLE_GRADES.length - 1}
          value={gradeIndex}
          onChange={handleSliderChange}
          marks={Object.fromEntries(
            AVAILABLE_GRADES.map((grade, idx) => [idx, { label: grade, style: { color: '#9ca3af', fontSize: '10px' } }])
          )}
          styles={{
            rail: { backgroundColor: '#374151', height: 4 },
            track: { backgroundColor: '#a855f7', height: 4 },
            handle: {
              backgroundColor: '#a855f7',
              borderColor: '#a855f7',
              height: 16,
              width: 16,
              marginTop: -6,
              opacity: 1
            }
          }}
          dotStyle={{
            backgroundColor: '#4b5563',
            borderColor: '#4b5563',
            height: 8,
            width: 8,
            bottom: -2
          }}
          activeDotStyle={{
            backgroundColor: '#a855f7',
            borderColor: '#a855f7'
          }}
        />
      </div>
    </div>
  );
}

