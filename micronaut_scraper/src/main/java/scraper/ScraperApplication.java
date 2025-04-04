package scraper;

import io.micronaut.runtime.Micronaut;

public class ScraperApplication {
    public static void main(String[] args) {
        Micronaut.run(ScraperApplication.class, args);
    }
}
// This is the main entry point for the Micronaut application. It initializes the application context and starts the server