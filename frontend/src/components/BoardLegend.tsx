import { HOLD_COLORS } from '../config/board';

interface LegendItemProps {
  color: string;
  label: string;
}

function LegendItem({ color, label }: LegendItemProps) {
  return (
    <div className="flex items-center gap-2">
      <div 
        className="w-4 h-4 rounded-full" 
        style={{ backgroundColor: color }}
      />
      <span className="text-sm text-gray-300">{label}</span>
    </div>
  );
}

export default function BoardLegend() {
  return (
    <div className="flex justify-center gap-6 mt-4">
      <LegendItem color={HOLD_COLORS.start} label="Start" />
      <LegendItem color={HOLD_COLORS.intermediate} label="Intermediate" />
      <LegendItem color={HOLD_COLORS.end} label="End" />
    </div>
  );
}

