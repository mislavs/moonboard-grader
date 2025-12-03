interface TabNavigationProps {
  activeMode: 'view' | 'create' | 'analytics';
  onModeChange: (mode: 'view' | 'create' | 'analytics') => void;
}

export default function TabNavigation({ activeMode, onModeChange }: TabNavigationProps) {
  const tabs = [
    { id: 'view' as const, label: 'Browse Problems' },
    { id: 'create' as const, label: 'Create Problem' },
    { id: 'analytics' as const, label: 'Board Analytics' },
  ];

  return (
    <div className="flex justify-center mb-8">
      <div className="inline-flex bg-gray-800 rounded-lg p-1 shadow-lg">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onModeChange(tab.id)}
            className={`px-8 py-3 rounded-md font-semibold transition-all duration-200 ${
              activeMode === tab.id
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-gray-300 hover:text-white hover:bg-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  );
}

