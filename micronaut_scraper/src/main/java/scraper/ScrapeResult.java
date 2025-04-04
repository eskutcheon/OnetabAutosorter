package scraper;

import java.util.Objects;

public class ScrapeResult {
    public String url;
    public String title = "";
    public String description = "";
    public String summary = "";

    public ScrapeResult(String url) {
        this.url = url;
    }

    public void appendToSummary(String text) {
        if (text != null && !text.isBlank()) {
            summary += text.strip() + "\n";
        }
    }

    @Override
    public boolean equals(Object o) {
        return (o instanceof ScrapeResult other) && Objects.equals(url, other.url);
    }

    @Override
    public int hashCode() {
        return Objects.hash(url);
    }
}
