import { useState, useEffect } from 'react';

let toastQueue = [];

export const useToast = () => {
  const [toasts, setToasts] = useState([]);

  useEffect(() => {
    if (toastQueue.length > 0) {
      setToasts([...toasts, ...toastQueue]);
      toastQueue = [];
    }
  }, [toasts]);

  const addToast = (toast) => {
    setToasts((prevToasts) => [...prevToasts, toast]);
  };

  return { toasts, addToast };
};

export const toast = (toast) => {
  toastQueue.push(toast);
};