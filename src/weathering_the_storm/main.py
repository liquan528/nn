import carla
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_carla_connection():
    """Test CARLA connection and run a simple simulation"""
    try:
        logging.info("Connecting to CARLA server at localhost:2000...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        world = client.get_world()
        logging.info("✅ Successfully connected to CARLA server!")

        # Get map info
        map_name = world.get_map().name
        logging.info(f"Current map: {map_name}")

        # Set up rainy weather
        weather = carla.WeatherParameters(
            cloudiness=100.0,
            precipitation=100.0,
            precipitation_deposits=100.0,
            wind_intensity=50.0,
            fog_density=20.0,
            wetness=100.0,
            sun_altitude_angle=45.0
        )
        world.set_weather(weather)
        logging.info("🌧️  Weather set to heavy rain!")

        # Spawn ego vehicle (motorcycle)
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.yamaha.yzf')
        spawn_points = world.get_map().get_spawn_points()

        if spawn_points:
            spawn_point = spawn_points[0]
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                logging.info(f"🏍️  Ego vehicle spawned: {vehicle_bp.id}")
                vehicle.set_autopilot(True)
                logging.info("🚗 Autopilot enabled!")

                # Setup traffic manager
                tm = client.get_trafficmanager(8000)
                tm.set_synchronous_mode(False)

                # Spawn some traffic vehicles
                vehicle_bps = blueprint_library.filter('vehicle.*')
                import random
                random.seed(42)

                spawned_vehicles = []
                for i in range(20):  # Spawn 20 vehicles
                    try:
                        bp = random.choice(vehicle_bps)
                        if bp.has_attribute('color'):
                            color = random.choice(bp.get_attribute('color').recommended_values)
                            bp.set_attribute('color', color)

                        spawn_pt = random.choice(spawn_points)
                        v = world.try_spawn_actor(bp, spawn_pt)
                        if v:
                            v.set_autopilot(True, tm.get_port())
                            spawned_vehicles.append(v)
                    except Exception as e:
                        continue

                logging.info(f"🚗 Spawned {len(spawned_vehicles)} traffic vehicles")

                # Run simulation for 30 seconds
                logging.info("\n" + "="*60)
                logging.info("🎬 Starting 30-second simulation...")
                logging.info("   Watch the CARLA window to see the simulation!")
                logging.info("="*60 + "\n")

                start_time = time.time()
                frame_count = 0

                while time.time() - start_time < 30:
                    world.tick()
                    frame_count += 1

                    if frame_count % 40 == 0:  # Every ~1 second at 40 FPS
                        elapsed = time.time() - start_time
                        logging.info(f"⏱️  Running... {elapsed:.1f}s / 30s | Frame: {frame_count}")

                logging.info("\n" + "="*60)
                logging.info(f"✅ Simulation completed! Total frames: {frame_count}")
                logging.info("="*60 + "\n")

                # Cleanup
                for v in spawned_vehicles:
                    if v.is_alive:
                        v.destroy()
                if vehicle.is_alive:
                    vehicle.destroy()
                logging.info("🧹 Cleanup completed!")

            else:
                logging.error("❌ Failed to spawn ego vehicle")
        else:
            logging.error("❌ No spawn points available")

    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_carla_connection()
