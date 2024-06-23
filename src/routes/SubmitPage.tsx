// SubmitPage.tsx
import { Typography, Container, Box, Divider } from "@mui/material";
import dayjs from "dayjs";
import { useEmail } from "../components/EmailProtienContext";

export default function SubmitPage() {
    const { email } = useEmail(); // Use the useEmail hook to get the email state
    const curDate = new Date();
    

    return (
        <Container>
            <Box sx={{ my: 4 }}>
                <Typography variant="h3">PLU Protien Research</Typography>
            </Box>

            <Box sx={{ my: 2 }}>
                <Divider />
                <Typography variant="h6">Email: {email}</Typography>
                <Divider />
            </Box>

            <Box sx={{ my: 2 }}>
                <Divider />
                <Typography variant="h6">Date: {curDate.toLocaleTimeString([], {year: 'numeric', month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'})}</Typography>
                <Divider />
            </Box>

            <Box sx={{ my: 2 }}>
                <Typography fontWeight={'bold'}>Thank you for using this tool, your results will be sent to the email provided shortly.</Typography>
            </Box>
        </Container>
    );
}
