import asyncio
import signal
from config import load_config, MachaConfig
from logger import setup_logger
from database import init_database
from scheduler import TaskScheduler


class Application:
    """Main application class to handle graceful shutdown."""

    def __init__(self):
        self.scheduler = None
        self.engine = None
        self.logger = None
        self.config = None
        self.shutdown_event = asyncio.Event()

    async def startup(self):
        """Initialize the application."""
        try:
            # Load and validate configuration
            self.config = load_config()
            self.logger = setup_logger(self.config)
            self.logger.info("Starting Macha application")
            self.logger.info(
                f"Configuration loaded with {len(self.config.tasks)} tasks"
            )

            # Initialize database
            self.engine = await init_database(self.config)
            self.logger.info("Database initialized")

            # Set up task scheduler
            self.scheduler = TaskScheduler(self.config, self.engine, self.logger)
            self.logger.info("Task scheduler initialized")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Application startup failed: {e}")
            else:
                print(f"Application startup failed: {e}")
            raise

    async def run(self):
        """Run the application."""
        try:
            # Start the scheduler
            await self.scheduler.start()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            raise

    async def shutdown(self):
        """Gracefully shutdown the application."""
        if self.logger:
            self.logger.info("Shutting down application")

        if self.scheduler:
            await self.scheduler.stop()

        if self.engine:
            await self.engine.dispose()

        if self.logger:
            self.logger.info("Application shutdown complete")


async def main():
    """Main entry point."""
    app = Application()

    # Setup signal handlers for graceful shutdown
    def signal_handler():
        print("\nReceived shutdown signal, gracefully shutting down...")
        app.shutdown_event.set()

    # Register signal handlers
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler())
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, lambda s, f: signal_handler())

    try:
        # Initialize application
        await app.startup()

        # Create tasks for running and waiting for shutdown
        scheduler_task = asyncio.create_task(app.scheduler.start())
        shutdown_task = asyncio.create_task(app.shutdown_event.wait())

        # Wait for either the scheduler to complete or shutdown signal
        done, pending = await asyncio.wait(
            [scheduler_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel any remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Application failed to start: {e}")
        if app.logger:
            app.logger.error(f"Application failed: {e}")
        return 1
    finally:
        # Ensure cleanup happens
        await app.shutdown()

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("Application interrupted by user")
        exit(0)
    except Exception as e:
        print(f"Application failed: {e}")
        exit(1)
