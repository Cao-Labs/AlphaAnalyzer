import { Box, Typography, Divider, Button, Container, Paper, Stack, CircularProgress } from "@mui/material";
import { ChangeEvent, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function HomePage() {
    const navigate = useNavigate();
    const [proteinFile, setProteinFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false); // Add loading state

    const handleSubmit = async () => {
        if (!proteinFile) {
            alert("Please select a file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", proteinFile);

        setLoading(true); // Set loading to true when submission starts

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData,
            });

            const contentType = response.headers.get("content-type");
            if (!response.ok) {
                const errorText = contentType?.includes("application/json") 
                    ? await response.json() 
                    : await response.text();
                throw new Error(`Server Error: ${JSON.stringify(errorText)}`);
            }

            const data = await response.json();
            console.log("Predictions:", data);
            navigate('/submit', { state: { results: data.predictions } });

        } catch (error) {
            console.error("Error submitting file:", error);
            alert(`Submission failed: ${error.message}`);
        } finally {
            setLoading(false); // Reset loading state
        }
    };

    const handleOnChangeFile = (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0] || null;
        setProteinFile(file);
    };

    return (
        <Container maxWidth="md" sx={{ py: 4 }}>
            <Paper elevation={3} sx={{ p: 4, mb: 4, textAlign: 'center', bgcolor: '#f5f5f5' }}>
                <Typography variant="h3" gutterBottom sx={{ fontWeight: 'bold', color: '#2c3e50' }}>
                    PLU Protein Research
                </Typography>
                <Divider sx={{ width: '100%', my: 2 }} />
            </Paper>

            <Paper elevation={3} sx={{ p: 4, mb: 4, bgcolor: '#ffffff' }}>
                <Typography variant="h5" gutterBottom sx={{ fontWeight: 'medium', color: '#34495e' }}>
                    Upload Your Protein File (CSV format)
                </Typography>
                <Stack spacing={3} sx={{ my: 2 }}>
                    <Box
                        component="form"
                        sx={{
                            '& .MuiTextField-root': { width: '100%' },
                        }}
                        noValidate
                        autoComplete="off"
                    >
                        <input 
                            type="file" 
                            onChange={handleOnChangeFile} 
                            style={{ display: 'none' }} 
                            id="file-upload"
                        />
                        <label htmlFor="file-upload">
                            <Button 
                                variant="outlined" 
                                component="span" 
                                fullWidth 
                                sx={{ py: 2, borderStyle: 'dashed', borderColor: '#bdc3c7', color: '#7f8c8d' }}
                            >
                                {proteinFile ? proteinFile.name : "Choose File"}
                            </Button>
                        </label>
                    </Box>

                    <Button 
                        fullWidth 
                        onClick={handleSubmit} 
                        variant="contained" 
                        disabled={loading} // Disable button when loading
                        sx={{ py: 2, bgcolor: '#3498db', '&:hover': { bgcolor: '#2980b9' } }}
                    >
                        {loading ? <CircularProgress size={24} sx={{ color: '#ffffff' }} /> : "Submit"}
                    </Button>
                </Stack>
            </Paper>

            <Paper elevation={3} sx={{ p: 4, bgcolor: '#f5f5f5' }}>
                <Stack spacing={4}>
                    <Box textAlign="center">
                        <Typography variant="h5" gutterBottom sx={{ fontWeight: 'medium', color: '#34495e' }}>
                            Citation
                        </Typography>
                        <Divider sx={{ width: '100%', my: 2 }} />
                        <Typography variant="body1" sx={{ color: '#7f8c8d' }}>
                            Dr. Cao Renzhi
                        </Typography>
                    </Box>

                    <Box textAlign="center">
                        <Typography variant="h5" gutterBottom sx={{ fontWeight: 'medium', color: '#34495e' }}>
                            Contact
                        </Typography>
                        <Divider sx={{ width: '100%', my: 2 }} />
                        <Typography variant="body1" sx={{ color: '#7f8c8d' }}>
                            If you have any questions, please contact:
                        </Typography>
                        <Typography variant="body1" sx={{ color: '#7f8c8d' }}>
                            Dr. Cao Renzhi
                        </Typography>
                        <Typography variant="body1" sx={{ color: '#7f8c8d' }}>
                            Department of Computer Science
                        </Typography>
                        <Typography variant="body1" sx={{ color: '#7f8c8d' }}>
                            Pacific Lutheran University
                        </Typography>
                    </Box>
                </Stack>
            </Paper>
        </Container>
    );
}