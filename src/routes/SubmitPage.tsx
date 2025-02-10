import { Box, Typography, Button, Container, Paper, Stack, List, ListItem, ListItemText } from "@mui/material";
import { useLocation, useNavigate } from "react-router-dom";

export default function SubmitPage() {
    const location = useLocation();
    const navigate = useNavigate();
    const results = location.state?.results || [];

    return (
        <Container maxWidth="md" sx={{ py: 4 }}>
            <Paper elevation={3} sx={{ p: 4, mb: 4, textAlign: 'center', bgcolor: '#f5f5f5' }}>
                <Typography variant="h3" gutterBottom sx={{ fontWeight: 'bold', color: '#2c3e50' }}>
                    Prediction Results
                </Typography>
            </Paper>

            <Paper elevation={3} sx={{ p: 4, mb: 4, bgcolor: '#ffffff' }}>
                <Typography variant="h5" gutterBottom sx={{ fontWeight: 'medium', color: '#34495e', mb: 3 }}>
                    Your Predictions
                </Typography>
                {results.length > 0 ? (
                    <List sx={{ maxHeight: 300, overflow: 'auto', border: '1px solid #e0e0e0', borderRadius: 1 }}>
                        {results.map((result: string, index: number) => (
                            <ListItem key={index} sx={{ borderBottom: '1px solid #e0e0e0', '&:last-child': { borderBottom: 'none' } }}>
                                <ListItemText primary={`Prediction ${index + 1}: ${result}`} />
                            </ListItem>
                        ))}
                    </List>
                ) : (
                    <Typography variant="body1" sx={{ color: '#7f8c8d', textAlign: 'center' }}>
                        No predictions available.
                    </Typography>
                )}
            </Paper>

            <Box textAlign="center">
                <Button 
                    variant="contained" 
                    onClick={() => navigate("/")} 
                    sx={{ py: 2, px: 4, bgcolor: '#3498db', '&:hover': { bgcolor: '#2980b9' } }}
                >
                    Back to Home
                </Button>
            </Box>
        </Container>
    );
}