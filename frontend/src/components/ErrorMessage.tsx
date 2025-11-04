interface ErrorMessageProps {
  title?: string;
  message: string;
  details?: string;
}

export default function ErrorMessage({ 
  title = 'Oops! Something went wrong',
  message,
  details = 'Please check your connection and try refreshing the page.'
}: ErrorMessageProps) {
  return (
    <div className="max-w-2xl mx-auto px-4">
      <div className="bg-red-900 border border-red-700 text-red-100 px-6 py-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">{title}</h3>
        <p>{message}</p>
        {details && (
          <p className="mt-2 text-sm text-red-200">{details}</p>
        )}
      </div>
    </div>
  );
}

