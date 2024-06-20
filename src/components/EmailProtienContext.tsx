// EmailContext.tsx
import React, { createContext, useContext, useState, ReactNode } from "react";

// Define the shape of the context value
interface EmailContextType {
    email: string;
    setEmail: (email: string) => void;
}

// Create the context with a default value
const EmailContext = createContext<EmailContextType>({
    email: "",
    setEmail: () => {}
});

// Define the props for the provider component
interface EmailProviderProps {
    children: ReactNode;
}

// Create a provider component
export const EmailProvider: React.FC<EmailProviderProps> = ({ children }) => {
    const [email, setEmail] = useState("");

    return (
        <EmailContext.Provider value={{ email, setEmail }}>
            {children}
        </EmailContext.Provider>
    );
};

// Custom hook to use the context
export const useEmail = () => {
    return useContext(EmailContext);
};
