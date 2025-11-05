interface DuplicateCheckResultProps {
  exists: boolean;
  problemName?: string;
  problemGrade?: string;
}

export default function DuplicateCheckResult({ 
  exists, 
  problemName, 
  problemGrade 
}: DuplicateCheckResultProps) {
  if (exists) {
    return (
      <div className="rounded-lg p-4 shadow-lg bg-yellow-900/50 border border-yellow-600">
        <div className="flex items-start gap-2">
          <span className="text-yellow-400 text-xl">⚠️</span>
          <div>
            <p className="text-yellow-300 font-semibold">Duplicate found:</p>
            <p className="text-white mt-1">
              {problemName} - {problemGrade}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg p-4 shadow-lg bg-green-900/50 border border-green-600">
      <div className="flex items-center gap-2">
        <span className="text-green-400 text-xl">✓</span>
        <p className="text-green-300 font-semibold">No duplicate found</p>
      </div>
    </div>
  );
}

