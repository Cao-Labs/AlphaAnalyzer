import { BrowserRouter, Route, Routes } from "react-router-dom";
import HomePage from "./routes/HomePage";
import SubmitPage from "./routes/SubmitPage";
import { EmailProvider } from "./components/EmailProtienContext";
function App() {

  return (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={
      <EmailProvider>
        <HomePage />
     </EmailProvider>}/>
      <Route path="/submit" element={
      <EmailProvider>
        <SubmitPage />
     </EmailProvider>
      }/>
    </Routes>
  </BrowserRouter>
  );
  
}

export default App;
