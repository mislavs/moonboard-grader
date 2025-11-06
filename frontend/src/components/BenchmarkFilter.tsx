interface BenchmarkFilterProps {
  benchmarkFilter: boolean | null;
  onFilterChange: (filter: boolean | null) => void;
}

export default function BenchmarkFilter({
  benchmarkFilter,
  onFilterChange,
}: BenchmarkFilterProps) {
  return (
    <div className="mb-4 pb-4 border-b border-gray-700">
      <div className="space-y-2">
        <label className="flex items-center text-gray-300 text-sm cursor-pointer hover:text-white transition-colors">
          <input
            type="radio"
            name="benchmark-filter"
            checked={benchmarkFilter === null}
            onChange={() => onFilterChange(null)}
            className="mr-2 cursor-pointer"
          />
          All Problems
        </label>
        <label className="flex items-center text-gray-300 text-sm cursor-pointer hover:text-white transition-colors">
          <input
            type="radio"
            name="benchmark-filter"
            checked={benchmarkFilter === true}
            onChange={() => onFilterChange(true)}
            className="mr-2 cursor-pointer"
          />
          Benchmarks Only
        </label>
        <label className="flex items-center text-gray-300 text-sm cursor-pointer hover:text-white transition-colors">
          <input
            type="radio"
            name="benchmark-filter"
            checked={benchmarkFilter === false}
            onChange={() => onFilterChange(false)}
            className="mr-2 cursor-pointer"
          />
          Non-Benchmarks Only
        </label>
      </div>
    </div>
  );
}

