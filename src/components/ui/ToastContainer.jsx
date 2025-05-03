import React from 'react';
import { useToast } from './use-toast';

const ToastContainer = () => {
  const { toasts } = useToast();

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((toast, index) => (
        <div
          key={index}
          className="bg-white rounded-lg shadow-lg p-4 transition-all transform"
          style={{ maxWidth: '350px' }}
        >
          {toast.title && <h4 className="font-semibold">{toast.title}</h4>}
          {toast.description && <p className="text-sm text-gray-600 mt-1">{toast.description}</p>}
        </div>
      ))}
    </div>
  );
};

export default ToastContainer;