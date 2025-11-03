import MoonBoard from './components/MoonBoard';
import type { Problem } from './types/problem';

// Hardcoded problem for testing
const sampleProblem: Problem = {
  name: "Fat Guy In A Little Suit",
  grade: "6B+",
  userGrade: "6B+",
  setbyId: "55C65799-AB21-496C-A34E-1CC4B3210B27",
  setby: "Kyle Knapp",
  method: "Feet follow hands",
  userRating: 4,
  repeats: 187,
  holdsetup: {
    description: "MoonBoard Masters 2017",
    holdsets: null,
    apiId: 15
  },
  isBenchmark: false,
  isMaster: false,
  upgraded: false,
  downgraded: false,
  moves: [
    {
      problemId: 305445,
      description: "J4",
      isStart: true,
      isEnd: false
    },
    {
      problemId: 305445,
      description: "G6",
      isStart: false,
      isEnd: false
    },
    {
      problemId: 305445,
      description: "F7",
      isStart: false,
      isEnd: false
    },
    {
      problemId: 305445,
      description: "B8",
      isStart: false,
      isEnd: false
    },
    {
      problemId: 305445,
      description: "B9",
      isStart: false,
      isEnd: false
    },
    {
      problemId: 305445,
      description: "A11",
      isStart: false,
      isEnd: false
    },
    {
      problemId: 305445,
      description: "D15",
      isStart: false,
      isEnd: false
    },
    {
      problemId: 305445,
      description: "A16",
      isStart: false,
      isEnd: false
    },
    {
      problemId: 305445,
      description: "F18",
      isStart: false,
      isEnd: true
    }
  ],
  holdsets: [
    {
      description: "Hold Set A",
      locations: null,
      apiId: 4
    },
    {
      description: "Hold Set B",
      locations: null,
      apiId: 5
    }
  ],
  hasBetaVideo: false,
  moonBoardConfigurationId: 1,
  apiId: 305445,
  dateInserted: "2017-10-20T11:27:40.393",
  dateUpdated: "2023-01-22T16:58:17.92",
  dateDeleted: null
};

function App() {
  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="text-center mb-8">
        <h1 className="text-5xl font-bold text-white mb-2">
          Moonboard Grader
        </h1>
        <p className="text-lg text-gray-400">
          AI-powered climbing grade prediction
        </p>
      </div>
      <MoonBoard problem={sampleProblem} />
    </div>
  )
}

export default App
