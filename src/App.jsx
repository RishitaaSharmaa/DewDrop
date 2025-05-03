import { Toaster } from "./components/ui/toaster";
import { Toaster as Sonner } from "./components/ui/sonner";
import { TooltipProvider } from "./components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { useState, useEffect } from "react";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import UserDashboard from "./pages/UserDashboard";
import AdminPanel from "./pages/AdminPanel";
import DevicePage from "./pages/DevicePage";
import HomePage from "./pages/HomePage";
import LoadingDroplet from "./components/LoadingDroplet";
import "./App.css";

const queryClient = new QueryClient();

const NavigationLoader = () => {
    const location = useLocation();
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        // Show loading state when navigation starts
        setIsLoading(true);

        // Hide loading state after a short delay to simulate page loading
        const timer = setTimeout(() => {
            setIsLoading(false);
        }, 800);

        return () => clearTimeout(timer);
    }, [location.pathname]);

    return isLoading ? <LoadingDroplet /> : null;
};

const AppRoutes = () => (
    <>
        <NavigationLoader />
        <Routes>
            <Route path="/DewDrop" element={<Index />} />
            <Route path="/home" element={<HomePage />} />
            <Route path="/device/:deviceId" element={<DevicePage />} />
            <Route path="/user-dashboard" element={<UserDashboard />} />
            <Route path="/admin-panel" element={<AdminPanel />} />
            <Route path="*" element={<NotFound />} />
        </Routes>
    </>
);

const App = () => (
    <QueryClientProvider client={queryClient}>
        <TooltipProvider>
            <Toaster />
            <Sonner />
            <BrowserRouter>
                <AppRoutes />
            </BrowserRouter>
        </TooltipProvider>
    </QueryClientProvider>
);

export default App; 